import os
import torch

from src.ml.api_client import MockAPIClient
from src.file_repository import LocalFileRepository
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

from typing import List
import hashlib

def sha256(data):
    return hashlib.sha256(data).digest()


    
class VideoTranslationPipeline:
    def __init__(self, config: TranslationPipelineConfig):
        self.config = config

        self.api_client = MockAPIClient('dontcare')

        self.file_repository = LocalFileRepository(
            self.config.public_id,
            base_directory=self.config.base_dir,
            s3_client=get_s3_client()
        )

        file = RemoteFile(
            file_path=self.config.source_video_path,
            name=self.config.name
        )
        self.file = self.file_repository.save_file(file, force=False)


        self.logger = Logger(directory=self.file_repository.directory)

        self.video_translation = VideoTranslation(source_file=file, public_id=self.config.public_id)
   
    
        
    def translate(self):
        stt_manager = SpeechToTextManager(self.config.public_id, self.api_client, self.file_repository, device="cuda", logger=self.logger)
        self.video_translation = stt_manager.extract_and_transcribe(self.video_translation, num_speakers=self.config.num_speakers, lang=self.config.source_lang)
        
        torch.cuda.empty_cache()
        translate_manager = TranslationManager(self.config.public_id, self.api_client, self.file_repository, device=self.config.device, logger=self.logger)
        self.video_translation = translate_manager.translate(self.video_translation, source_lang=self.config.source_lang, target_lang=self.config.target_lang)
        
        
        torch.cuda.empty_cache()
        tts_manager = TextToSpeechManager(self.config.public_id, self.api_client, self.file_repository, device=self.config.device, logger=self.logger, tts_sample_rate=24000)
        self.video_translation = tts_manager.synthesize(self.video_translation, source_lang=self.config.source_lang, target_lang=self.config.target_lang, voice_conv=True, merge_pipeline="stretch_whole", enhance=True)
        return self.video_translation


class ChangeManager:
    def __init__(self, video_translation_pipeline: VideoTranslationPipeline, video_translation: VideoTranslation):
        self.video_translation_pipeline = video_translation_pipeline
        self.video_translation = video_translation

    
    def compare_translations(self, new_text_transltaions : List[str]):
        updates = []
        for i, new_text in enumerate(new_text_transltaions):
            if new_text != self.video_translation.translated_texts[i].text:
                updates.append(TraslationUpdate(index=i, text=new_text))
        return updates
    
    def apply_update_translations(self, updates: List[TraslationUpdate]):
        for update in updates:
            self.video_translation.translated_texts[update.index].text = update.text
        json_segments = [{"translation": seg.translation, "text": seg.text} for seg in self.video_translation.translated_texts]
        self.video_translation_pipeline.logger.log_json(file_name="translations.json", data=json_segments)
        
        tts_manager = TextToSpeechManager(self.video_translation_pipeline.config.public_id, self.video_translation_pipeline.api_client, self.video_translation_pipeline.file_repository, device=self.video_translation_pipeline.config.device, logger=self.video_translation_pipeline.logger, tts_sample_rate=24000)
        for update in updates:
            tts_manager.synthesize_segment(self.video_translation.translated_texts[update.index], self.video_translation, source_lang=self.video_translation_pipeline.config.source_lang, target_lang=self.video_translation_pipeline.config.target_lang, voice_conv=True)
        tts_manager.clear_result_video(self.video_translation_pipeline.file_repository.directory + "/resulted_video.mp4")
        
        tts_manager.synthesize(self.video_translation, source_lang=self.video_translation_pipeline.config.source_lang, target_lang=self.video_translation_pipeline.config.target_lang, voice_conv=True, merge_pipeline="stretch_whole", enhance=True)
        
        
    