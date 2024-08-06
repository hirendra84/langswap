import pandas as pd

import spacy

from logging import getLogger

from src.ml.api_client import APIClient
from src.pipeline_models.enums import ProcessStatus
from src.ml.ffmpeg import FFmpegClient
from src.file_repository import FileRepository
from src.pipeline_models.models import RemoteFile
from src.pipeline_models.models import TextedSegment, VideoTranslation
from src.ml.speech_to_text_service import asr_client
from src.ml.speech_to_text_service.asr_client import ASRClient
from src.ml.speech_to_text_service.vad_client import VadClient
from src.ml.text_to_speech_service.demucs_client import DemucsClient

logger = getLogger(__name__)


class SpeechToTextManager:
    public_id: str

    _asr_client: ASRClient
    _api_client: APIClient
    _file_repository: FileRepository
    sample_rate: int = 16_000

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository):
        self.public_id = public_id
        self._asr_client = ASRClient("PMKV2A3076HULPET7XZSF7IITZP5H8SWICSCFI3L")
        self._api_client = api_client
        self._file_repository = file_repository

    def _resample_audio(self, audio_file: RemoteFile) -> RemoteFile:
        resampled_audio_file = self._file_repository.get_file(f'{audio_file.name}_resampled_{self.sample_rate}')
        (FFmpegClient()
         .resample_audio(audio_file.file_path,
                         resampled_audio_file.file_path,
                         sample_rate=self.sample_rate))
        vad_filtered_audio_file = self._file_repository.get_file(f'{resampled_audio_file.name}_vad')

        vad_filtered_audio_file.file_path = VadClient().vad_filter(
            resampled_audio_file.file_path,
            vad_filtered_audio_file.file_path,
            self.sample_rate)
        return vad_filtered_audio_file

    def extract_and_transcribe(self, video_translation: VideoTranslation, lang: str) -> VideoTranslation:

        video_name = self._file_repository.materialize_file(video_translation.source_file)

        audio_file = self._extract_audio(video_name.file_path)
        audio_file = self._file_repository.save_file(audio_file)

        self._api_client.update_video(self.public_id,
                                      video_translation,
                                      progress=10,
                                      status=ProcessStatus.in_progress)
        
        background_paths = DemucsClient().separate(audio_file.file_path, self._file_repository.subdir('background_files'))

        background_files = {name: self._file_repository.save_file(
            RemoteFile(name=name,
                       file_path=path),
            force=True
        ) for path, name in background_paths}

        # transcribe only according to the vocals
        vocal_file = background_files["vocals.wav"]
        transcription = self._asr_client.transcribe(vocal_file.s3_url, lang=lang)

        self._api_client.update_video(self.public_id,
                                      video_translation,
                                      progress=30,
                                      status=ProcessStatus.in_progress)

        segments = self._remap_sentences(transcription.word_timestamps, lang=lang)

        return VideoTranslation(
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=audio_file,
            background_audio=background_files,
            recognized_texts=segments,
            processed_video=video_translation.processed_video,
        )

    def _download_video(self, file: RemoteFile):
        return self._file_repository.materialize_file(file)

    def _extract_audio(self, video_file_path, audio_file_name='extracted_audio') -> RemoteFile:
        ffmpeg_client = FFmpegClient()
        output_file = self._file_repository.get_file(audio_file_name)
        out, err = ffmpeg_client.extract_audio(video_file_path,
                                               output_file.file_path,
                                               time_limit=60)
        logger.info(out)
        logger.error(err)

        return output_file

    def _remap_sentences(self, transcription: list[asr_client.WordTimestamp], lang: str) -> list[TextedSegment]:
        def load_spacy_model(language='xx') -> spacy.Language:
            spacy_languages = {
                'en': "en_core_web_sm",
                'ru': "ru_core_news_sm",
                'fr': "fr_core_news_sm",
                'zh': "zh_core_web_sm",
                "de": "de_core_news_sm",
                "nl": "nl_core_news_sm",
                "pl": "pl_core_news_sm",
                "es": "es_core_news_sm",
                "xx": "xx_sent_ud_sm"  # multilingual model
            }
            selected_model = spacy_languages[language]
            try:
                nlp_ = spacy.load(selected_model)
            except OSError:
                spacy.cli.download(selected_model)
                nlp_ = spacy.load(selected_model)
            return nlp_

        nlp = load_spacy_model()
        plain_text = ' '.join(t.word for t in transcription)
        doc = nlp(plain_text)

        sent_bounds = [s[0].idx for s in doc.sents]

        transcription_dict = [{
            'word': t.word,
            'start': t.start,
            'end': t.end,
        } for t in transcription]

        df_words = pd.DataFrame(transcription_dict)

        df_words['text'] = df_words.word
        df_words['len'] = df_words.text.apply(len)
        df_words['end_pos'] = (df_words['len'] + 1).cumsum()
        df_words['start_pos'] = df_words['end_pos'].shift(1, fill_value=0)

        for i, x in enumerate(sent_bounds):
            df_words.loc[df_words['end_pos'] > x, 'sent'] = i

        df_words.sent = df_words.sent.astype(int)        
        entries = []
        for i in df_words.sent.unique():
            slc = df_words.loc[df_words.sent == i]

            entries.append({"text": ' '.join(slc.text.to_list()), "start": slc.start.min(), "end": slc.end.max()})   

        # algorithm to merge sentences according to the threshold
        pairs = []
        threshold = 0.5
        prev_end = entries[0]['end']
        start_idx = entries[0]['start']
        cur_pair = [entries[0]]

        for cur_sample in entries[1:]:
            # если пауза достаточно длинная и при этом мы собрали уже около 5 секунд
            if ((cur_sample['start'] - prev_end) >= threshold and (prev_end - start_idx >= 5)) or (prev_end - start_idx >= 30): 
                pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    )
                )                   
                cur_pair = [cur_sample]
                prev_end = cur_sample['end']
                start_idx = cur_sample['start']
            else:
                cur_pair.append(cur_sample)       
                prev_end = cur_sample['end']

        if cur_pair:
            pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    )
                )   
        return pairs

