import os
import torch
import torchaudio

from src.ml.api_client import MockAPIClient
from src.file_repository import LocalFileRepository
from src.pipeline_models.enums import ProcessStatus
from src.pipeline_models.models import RemoteFile
from src.pipeline_models.models import VideoTranslation
from src.pipeline_models.models import TranslationPipelineConfig
from src.pipeline_models.models import TraslationUpdate
from src.settings import BASE_WORKING_DIR
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.s3_client import get_s3_client
from src.utils.logging import Logger
from src.ml.video_dubbing_manager import VideoDubbingManager
from src.ml.text_to_speech_service.demucs_client import DemucsClient
from src.ml.ffmpeg import FFmpegClient

from typing import List


    
class VideoTranslationPipeline:
    def __init__(self, config: TranslationPipelineConfig):
        self.config = config

        self._api_client = MockAPIClient('dontcare')

        self._file_repository = LocalFileRepository(
            self.config.public_id,
            base_directory=self.config.base_dir,
            s3_client=get_s3_client()
        )

        file = RemoteFile(
            file_path=self.config.source_video_path,
            name=self.config.name
        )
        self.file = self._file_repository.save_file(file, force=False)


        self.logger = Logger(directory=self._file_repository.directory)

        self.video_translation = VideoTranslation(source_file=file, public_id=self.config.public_id)
        
        self.audio_extensions = ["mp3", "wav", "MP3"]
   
    def _generate_asr(self):
        stt_manager = SpeechToTextManager(self.config.public_id, self._api_client, self._file_repository, device=self.config.device, logger=self.logger)
        self.video_translation = stt_manager.extract_and_transcribe(self.video_translation, num_speakers=self.config.num_speakers, lang=self.config.source_lang)

    def _generate_translation(self):
        torch.cuda.empty_cache()
        translate_manager = TranslationManager(self.config.public_id, self._api_client, self._file_repository, device=self.config.device, logger=self.logger)
        self.video_translation = translate_manager.translate(self.video_translation, source_lang=self.config.source_lang, target_lang=self.config.target_lang)
        
    def _generate_speech(self):
        torch.cuda.empty_cache()
        tts_manager = TextToSpeechManager(self.config.public_id, self._api_client, self._file_repository, device=self.config.device, logger=self.logger, tts_sample_rate=24000)
        self.video_translation = tts_manager.synthesize(self.video_translation, source_lang=self.config.source_lang, target_lang=self.config.target_lang, voice_conv=True, enhance=True)
    
    def _merge(self, merge_pipeline="stretch_whole"):
        video_dubbing_manager = VideoDubbingManager(self._file_repository, self.logger)
        vocals_audio = self.video_translation.background_audio["vocals.wav"]
        
        if merge_pipeline == "pause_based":
            generated_audio, generated_sr = video_dubbing_manager.merge_timestamps_pause_based(
                self.video_translation,
                vocals_audio
            )
        elif merge_pipeline == "stretch_whole":
            generated_audio, generated_sr = video_dubbing_manager.merge_timestamps_stretch_whole(
                self.video_translation,
                vocals_audio
            ) 
        elif merge_pipeline == "speedup":
            generated_audio, generated_sr = video_dubbing_manager.merge_timestamps_speedup(
                self.video_translation,
                vocals_audio
            )

        # TODO: save correctly if need on the s3
        styled_audio = self._file_repository.get_file("styled_full_audio.wav")
        torchaudio.save(styled_audio.file_path, generated_audio, generated_sr)

        audio_backgrounds = {
            name: remote_file
            for name, remote_file in
            self.video_translation.background_audio.items()
        }

        self.logger.file_logger.info("Step: merge backgrounds back")
        merged_background_audio, save_sr = DemucsClient().merge_background(
                    styled_audio.file_path,
                    audio_backgrounds,
        )

        self.logger.file_logger.info("Step: merge the video with audio")
        result_audio = self._file_repository.get_file("merged_background_audio.wav")
        torchaudio.save(result_audio.file_path, merged_background_audio, save_sr)

        resulted_video = self._file_repository.get_file("resulted_video.mp4")
        source_video = self.video_translation.source_file.file_path

        base, extension = os.path.splitext(self.video_translation.source_file.file_path)

        if extension not in self.audio_extensions:
            FFmpegClient().replace_audio(source_video,
                                        result_audio.file_path,
                                        resulted_video.file_path)
            self._file_repository.save_file(resulted_video)

        new_video_translation = VideoTranslation(
            public_id=self.video_translation.public_id,
            source_file=self.video_translation.source_file,
            extracted_audio=self.video_translation.extracted_audio,
            background_audio=self.video_translation.background_audio,
            vad_filtered_audio=self.video_translation.vad_filtered_audio,
            recognized_texts=self.video_translation.recognized_texts,
            translated_texts=self.video_translation.translated_texts,
            processed_video=resulted_video
        )

        self._api_client.update_video(new_video_translation.public_id,
                                      new_video_translation,
                                      progress=10,
                                      status=ProcessStatus.done)
        return new_video_translation
    
    def translate_video(self):
        self._generate_asr()
        self._generate_translation()
        self._generate_speech()
        self.video_translation = self._merge("stretch_whole")
        torch.cuda.empty_cache()
        return self.video_translation


class ChangeManager:
    def __init__(self, video_translation_pipeline: VideoTranslationPipeline, video_translation: VideoTranslation):
        self.video_translation_pipeline = video_translation_pipeline
        self.video_translation = video_translation

    
    def compare_translations(self, new_text_transltaions : List[str]):
        updates = []
        for i, new_text in enumerate(new_text_transltaions):
            if new_text != self.video_translation.translated_texts[i].translation:
                updates.append(TraslationUpdate(index=i, text=new_text))
        return updates
    
    def apply_update_translations(self, updates: List[TraslationUpdate]):
        for update in updates:
            self.video_translation.translated_texts[update.index].translation = update.text
        json_segments = [{"translation": seg.translation, "text": seg.text} for seg in self.video_translation.translated_texts]
        self.video_translation_pipeline.logger.log_json(file_name="translations.json", data=json_segments)
        
        tts_manager = TextToSpeechManager(self.video_translation_pipeline.config.public_id, self.video_translation_pipeline.api_client, self.video_translation_pipeline.file_repository, device=self.video_translation_pipeline.config.device, logger=self.video_translation_pipeline.logger, tts_sample_rate=24000)
        for update in updates:
            tts_manager.synthesize_segment(self.video_translation.translated_texts[update.index], self.video_translation, source_lang=self.video_translation_pipeline.config.source_lang, target_lang=self.video_translation_pipeline.config.target_lang, voice_conv=True)
        tts_manager.clear_result_video(self.video_translation_pipeline.file_repository.directory + "/resulted_video.mp4")
        
        self.video_translation = tts_manager.synthesize(self.video_translation, source_lang=self.video_translation_pipeline.config.source_lang, target_lang=self.video_translation_pipeline.config.target_lang, voice_conv=True, merge_pipeline="stretch_whole", enhance=True)
        
        
    