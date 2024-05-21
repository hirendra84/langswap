import io
import os.path

import pandas as pd

import spacy

from logging import getLogger

import requests

from src.ffmpeg import FFmpegClient
from src.pipeline_models import TextedSegment, VideoTranslation
from src.speech_to_text_service import asr_client
from src.speech_to_text_service.asr_client import ASRClient
from src.speech_to_text_service.vad_client import VadClient
from src.utils import upload_file_to_s3

logger = getLogger(__name__)


class SpeechToTextManager:
    directory: str
    public_id: str

    _asr_client: ASRClient
    sample_rate: int = 16_000

    def __init__(self, public_id: str, directory: str = None):
        self.public_id = public_id
        self._asr_client = ASRClient("PMKV2A3076HULPET7XZSF7IITZP5H8SWICSCFI3L")

        if directory is None:
            directory = os.path.join('/Users/nikolaypakhtusov/', 'data', public_id)

        self.directory = directory

    def _resample_audio(self, audio_file_name: str) -> str:

        resampled_audio_path = os.path.join(self.directory, f'{audio_file_name}_resampled_{self.sample_rate}')
        (FFmpegClient()
         .resample_audio(os.path.join(self.directory, audio_file_name),
                         resampled_audio_path,
                         sample_rate=self.sample_rate))
        vad_filtered_audio_path = f'{resampled_audio_path}_vad'
        vad_filtered_audio_path = VadClient().vad_filter(resampled_audio_path,
                                                         vad_filtered_audio_path,
                                                         self.sample_rate)
        return vad_filtered_audio_path

    def extract_and_transcribe(self, video_translation: VideoTranslation) -> VideoTranslation:
        os.makedirs(self.directory, exist_ok=True)

        video_name = self._download_video(video_translation.source_url)
        audio_file_name = self._extract_audio(video_name)

        with open(os.path.join(self.directory, audio_file_name), 'rb') as f:
            extracted_audio_link = upload_file_to_s3(io.BytesIO(f.read()), self.public_id)

        vad_filtered_audio_path = self._resample_audio(audio_file_name)

        with open(vad_filtered_audio_path, 'rb') as f:
            vad_filtered_audio_link = upload_file_to_s3(io.BytesIO(f.read()), self.public_id)

        transcription = self._asr_client.transcribe(vad_filtered_audio_link)

        segments = self._remap_sentences(transcription.word_timestamps)

        return VideoTranslation(
            source_url=video_translation.source_url,
            extracted_audio_url=extracted_audio_link,
            vad_filtered_audio_url=vad_filtered_audio_link,
            recognized_texts=segments,
            processed_video=video_translation.processed_video,
        )

    def _download_video(self, url: str,
                        file_name: str = 'downloaded_video',
                        chunk_size: int = 8192):
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(os.path.join(self.directory, file_name), 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        return file_name

    def _extract_audio(self, video_file_name, audio_file_name='extracted_audio') -> str:
        ffmpeg_client = FFmpegClient()
        out, err = ffmpeg_client.extract_audio(os.path.join(self.directory, video_file_name),
                                               os.path.join(self.directory, audio_file_name),
                                               time_limit=60)
        logger.info(out)
        logger.error(err)

        return audio_file_name

    def _remap_sentences(self, transcription: list[asr_client.WordTimestamp]) -> list[TextedSegment]:
        def load_spacy_model(language='xx') -> spacy.Language:
            spacy_languages = {
                'en': "en_core_web_sm",
                'ru': "ru_core_news_sm",
                'fr': "fr_core_news_sm",
                'zh': "zh_core_web_sm",
                "de": "de_core_news_sm",
                "nl": "nl_core_news_sm",
                "pl": "pl_core_news_sm",
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
        sentences = []
        for i in df_words.sent.unique():
            slc = df_words.loc[df_words.sent == i]
            entry = TextedSegment(
                text=' '.join(slc.text.to_list()),
                start=slc.start.min(),
                end=slc.end.max(),
            )
            sentences.append(entry)
        return sentences
