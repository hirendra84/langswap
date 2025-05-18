import os
from coqui.TTS.tts.layers.xtts import xtts_manager
import torch
import torchaudio

import src.model_config
from src.pipeline_models.models import RemoteFile
from src.pipeline_models.models import VideoTranslation
from src.pipeline_models.models import TranslationPipelineConfig, save_config_to_json, load_config_from_json
from src.pipeline_models.models import TraslationUpdate
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.logging import Logger
from src.ml.video_dubbing_manager import VideoDubbingManager
from src.ml.text_to_speech_service.demucs_client import DemucsClient
from src.ml.ffmpeg import FFmpegClient
from src.utils.ml_processing.lang2code_mapper import map_language_to_code

from typing import List


    
class VideoTranslationPipeline:
    def __init__(self, config: TranslationPipelineConfig, file_repository):
        self.config = config
        
        self._file_repository = file_repository
        config_file = os.path.join(self._file_repository.directory, "config.json")
        save_config_to_json(config, config_file)
        config_file = self._file_repository.get_file("config.json")
        self._file_repository.save_file(config_file)
        

        file = RemoteFile(
            file_path=self.config.source_video_path,
            name=self.config.public_id
        )

        self.logger = Logger(directory=self._file_repository.directory)

        self.video_translation = VideoTranslation(source_file=file, public_id=self.config.public_id)
        
        self.audio_extensions = ["mp3", "wav"]
   
    def _generate_asr(self):
        print("initializing ASR manager")
        stt_manager = SpeechToTextManager(language=self.config.source_lang, public_id=self.config.public_id, file_repository=self._file_repository, device=self.config.device, logger=self.logger)
        print(f"Transcribing audio file: {self.video_translation.source_file.file_path} with language: {self.config.source_lang}")
        self.video_translation = stt_manager.extract_and_transcribe(self.video_translation, num_speakers=self.config.num_speakers)
        if self.config.source_lang != None:
            self.config.source_lang = map_language_to_code(self.video_translation.source_lang_code, system="reverse_from_whisper")
        

    def _generate_translation(self):
        torch.cuda.empty_cache()
        translate_manager = TranslationManager(self.config.public_id, 
                                               self._file_repository, 
                                               device=self.config.device, 
                                               logger=self.logger)
        self.video_translation = translate_manager.translate(self.video_translation, source_lang=self.config.source_lang, target_lang=self.config.target_lang)

    def _generate_speech(self):
        torch.cuda.empty_cache()
        tts_manager = TextToSpeechManager(self.config.public_id, 
                                          self._file_repository, 
                                          device=self.config.device, 
                                          tts_name = self.config.tts_model,
                                          logger=self.logger, 
                                        #   tts_sample_rate=44100, 
                                          eleven_api_token=self.config.eleven_api_token)
        self.video_translation = tts_manager.synthesize(self.video_translation, 
                                                        source_lang=self.config.source_lang, 
                                                        target_lang=self.config.target_lang, 
                                                        voice_conv=self.config.voice_conv, 
                                                        enhance=True)
    
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
        else:
            raise ValueError(f"Invalid merge pipeline: {merge_pipeline}. Please use one of the following: pause_based, stretch_whole, speedup.")

        # TODO: save correctly if need on the s3
        styled_audio = self._file_repository.get_file("styled_full_audio.wav")
        torchaudio.save(styled_audio.file_path, generated_audio, generated_sr)
        self._file_repository.save_file(styled_audio)

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
        self._file_repository.save_file(result_audio)

        resulted_video = self._file_repository.get_file("resulted_video.mp4")
        source_video = self.video_translation.source_file.file_path

        base, extension = os.path.splitext(self.video_translation.source_file.file_path)

        if extension.lower() not in self.audio_extensions:
            FFmpegClient().replace_audio(source_video,
                                        result_audio.file_path,
                                        resulted_video.file_path,
                                        )
            self._file_repository.save_file(resulted_video)

        if self.config.watermark:
            self.logger.file_logger.info("Step: add watermark to the video")
            FFmpegClient().add_watermark(resulted_video.file_path,
                                          resulted_video.file_path)
            self._file_repository.save_file(resulted_video, force=True)
            self.logger.file_logger.info("Step: watermark added to the video")
            

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

        return new_video_translation
    
    def translate_video(self):
        self._generate_asr()
        self._generate_translation()
        self._generate_speech()
        self.video_translation = self._merge(self.config.dubbing_algo)
        torch.cuda.empty_cache()

        return self.video_translation

    def generate_srt_files(self):
        """
        Generate SRT files for both the original transcript and the translation.
        Returns the RemoteFile objects for both files.
        """
        # Function to convert timestamp to SRT format (HH:MM:SS,mmm)
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
        
        # Generate SRT for original transcript
        source_srt_content = ""
        for i, segment in enumerate(self.video_translation.recognized_texts):
            source_srt_content += f"{i+1}\n"
            source_srt_content += f"{format_time(segment.start)} --> {format_time(segment.end)}\n"
            source_srt_content += f"{segment.text}\n\n"
        
        # Generate SRT for translated transcript
        translated_srt_content = ""
        for i, segment in enumerate(self.video_translation.translated_texts):
            translated_srt_content += f"{i+1}\n"
            translated_srt_content += f"{format_time(segment.start)} --> {format_time(segment.end)}\n"
            translated_srt_content += f"{segment.translation}\n\n"
        
        # Save source SRT file
        source_srt_file = self._file_repository.get_file("source_transcript.srt")
        with open(source_srt_file.file_path, 'w', encoding='utf-8') as f:
            f.write(source_srt_content)
        
        # Save translated SRT file
        translated_srt_file = self._file_repository.get_file("translated_transcript.srt")
        with open(translated_srt_file.file_path, 'w', encoding='utf-8') as f:
            f.write(translated_srt_content)
        
        # Upload the files to S3
        source_srt_file = self._file_repository.save_file(source_srt_file)
        translated_srt_file = self._file_repository.save_file(translated_srt_file)
        
        return source_srt_file, translated_srt_file


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
        
        tts_manager = TextToSpeechManager(self.video_translation_pipeline.config.public_id, self.video_translation_pipeline._file_repository, device=self.video_translation_pipeline.config.device, logger=self.video_translation_pipeline.logger, tts_sample_rate=24000)
        vocals_path = self.video_translation.background_audio["vocals.wav"]
        for update in updates:
            tts_manager.synthesize_segment(self.video_translation.translated_texts[update.index], target_lang=self.video_translation_pipeline.config.target_lang, vocals_path=vocals_path, voice_conv=self.video_translation_pipeline.config.voice_conv)
        tts_manager.clear_result_video(self.video_translation_pipeline._file_repository.directory + "/resulted_video.mp4")
        
        self.video_translation = self.video_translation_pipeline._merge(self.video_translation_pipeline.config.dubbing_algo)
        return self.video_translation
        
        
    