import os
import json
from typing import List, Dict
from logging import getLogger

from src.ml.ffmpeg import FFmpegClient
from src.file_repository import FileRepository
from src.pipeline_models.models import RemoteFile
from src.pipeline_models.models import TextedSegment, VideoTranslation
from src.ml.speech_to_text_service.asr_client import ASRClient, ASRX
from src.ml.speech_to_text_service.vad_client import VadClient
from src.ml.text_to_speech_service.demucs_client import DemucsClient

logger = getLogger(__name__)


class SpeechToTextManager:
    public_id: str

    _asr_client: ASRClient
    _file_repository: FileRepository
    sample_rate: int = 16_000
    lang: str

    def __init__(self, public_id: str, file_repository: FileRepository, device, logger):
        self.public_id = public_id
        self._asr_client = ASRX(device=device)
        self._file_repository = file_repository

        self.logger = logger

        self.audio_extensions = ["mp3", "wav", "MP3"]

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

    def extract_and_transcribe(self, video_translation: VideoTranslation, num_speakers: int = None, lang: str = None) -> VideoTranslation:
        if num_speakers is not None:
            num_speakers = int(num_speakers)
        # video_name = self._file_repository.materialize_file(video_translation.source_file)
        base, extension = os.path.splitext(video_translation.source_file.file_path)

        if extension in self.audio_extensions:
            audio_file = video_translation.source_file.file_path
        else:
            audio_file = self._extract_audio(video_translation.source_file.file_path)
            audio_file = self._file_repository.save_file(audio_file)
            
        
        self.logger.file_logger.info('Step: Demucs separation')
        background_paths = DemucsClient().separate(audio_file.file_path, self._file_repository.subdir('background_files'))
        self._file_repository.save_dir(self._file_repository.subdir('background_files'))

        background_files = {
            name: path for path, name in background_paths
        }

        self.logger.file_logger.info(f'Step: Transcribing')
        vocal_file = background_files["vocals.wav"]
        source_lang_code = None
        # transcribe
        file_name = "raw_transcribed_info.json"
        log_text = os.path.join(self._file_repository.directory, file_name)

        lang_file_name = "lang_detect_info.json"
        log_lang = os.path.join(self._file_repository.directory, lang_file_name)

        if os.path.exists(log_text) and os.path.exists(log_lang):
            self.logger.file_logger.info(f'Getting info from transcribed samples')
            with open(log_text, encoding="utf-8") as f:
                json_segments = json.load(f)
                
                detect_lang = json.load(open(log_lang, encoding="utf-8"))
                source_lang_code = detect_lang["detected_language"]
        else:
            self.logger.file_logger.info(f'Loading the model on the disk')
            with self._asr_client as asr_client:
                #asr_client.load_models()
                transcription = asr_client.transcribe(vocal_file, num_speakers=num_speakers, lang=lang)
                source_lang_code = transcription.detected_language
                json_segments = [{"text": seg.text, "start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in transcription.segments]
                detect_lang = {"detected_language": source_lang_code}
                self.logger.log_json(file_name=lang_file_name, data=detect_lang)
                self.logger.log_json(file_name=file_name, data=json_segments)

                local_log_text = self._file_repository.get_file(file_name)
                self._file_repository.save_file(local_log_text)

                local_log_lang = self._file_repository.get_file(lang_file_name)
                self._file_repository.save_file(local_log_lang)

        # split in sentences for pauses     
        file_name = "splitted_sentences_pauses.json"
        log_text = os.path.join(self._file_repository.directory, file_name)

        if os.path.exists(log_text):
            self.logger.file_logger.info(f'Getting info from already splitted samples')
            with open(log_text, encoding="utf-8") as f:
                json_segments = json.load(f)
            
            segments = []

            for seg in json_segments:
                segments.append(TextedSegment(
                    text=seg['text'],
                    start=seg['start'],
                    end=seg['end'],
                    speaker=seg['speaker'])
                )
        else:
            segments = self._remap_pauses(json_segments)
            json_segments = [{"text": seg.text, "start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in segments]
            self.logger.log_json(file_name=file_name, data=json_segments)
            local_log_text = self._file_repository.get_file(file_name)
            self._file_repository.save_file(local_log_text)

        return VideoTranslation(
            source_lang_code=source_lang_code,
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=audio_file,
            background_audio=background_files,
            recognized_texts=segments,
            processed_video=video_translation.processed_video,
        )

    def _download_video(self, file: RemoteFile):
        return self._file_repository.materialize_file(file)

    def _extract_audio(self, video_file_path, audio_file_name='extracted_audio.wav') -> RemoteFile:
        ffmpeg_client = FFmpegClient()
        output_file = self._file_repository.get_file(audio_file_name)
        out, err = ffmpeg_client.extract_audio(video_file_path,
                                               output_file.file_path,
                                               time_limit=60)
        logger.info(out)
        logger.error(err)

        return output_file

    def _remap_pauses(self, entries: List[Dict], pause_threshold=0.9, max_length=5, min_length=3): 
        # algorithm to merge sentences according to the threshold
        pairs = []
        prev_end = entries[0]['end']
        start_idx = entries[0]['start']
        cur_pair = [entries[0]]
        cur_speaker = entries[0]['speaker']
        check_is_text = lambda x: len([i for i in x if i.isalpha()]) > 0

        for cur_sample in entries[1:]:
            if not check_is_text(cur_sample['text']):
                continue
            conditions = [
                # pause is long enough and we have collected more than five seconds 
                # change for ElevenLabs that does not condition on one audio sample
                # ((cur_sample['start'] - prev_end) >= pause_threshold and (prev_end - start_idx >= min_length)),
                # (cur_sample['start'] - prev_end) >= pause_threshold,
                # we have collected max length of audio 
                (prev_end - start_idx > max_length),
                
                # speaker changed
                cur_speaker != cur_sample['speaker'],
                # not (prev_end - start_idx < 0.6)
                ]
            # print(f"what conditions are fullfilled {conditions}")
            # если пауза достаточно длинная и при этом мы собрали уже около 5 секунд или уже собрали больше 15 секунд
            if any(conditions):
                pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    speaker=cur_speaker
                    )
                )               
                cur_pair = [cur_sample]
                prev_end = cur_sample['end']
                start_idx = cur_sample['start']
                
                cur_speaker = cur_sample['speaker']
            else:
                cur_pair.append(cur_sample)       
                prev_end = cur_sample['end']

        if cur_pair:
            pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    speaker=cur_speaker
                    )
                )   
        # last sample use case - sometimes it is too short and should be better merged with a previous one
        last_sample = pairs[-1]
        if last_sample.end - last_sample.start < 3 and len(pairs) > 1:
            pairs[-2].text = pairs[-2].text + " " + last_sample.text
            pairs[-2].end = last_sample.end
        return pairs 