import torchaudio
import os

from logging import getLogger

from src.file_repository import FileRepository
from src.pipeline_models.models import VideoTranslation
from src.pipeline_models.models import TranslatedTextedSegment
from src.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from src.ml.text_to_speech_service.tts_client import TTSClient
from src.ml.text_to_speech_service.tts_xtts_client import XTTSClient
from src.ml.text_to_speech_service.tts_f5_client import FlowClient
from src.ml.text_to_speech_service.tts_eleven_client import ElevenTTSClient
from src.ml.text_to_speech_service.tts_fish_speech_client import FishSpeechClient
from src.ml.text_to_speech_service.voice_converter import VoiceToneConverter
from src.utils.ml_processing.lang2code_mapper import map_language_to_code


logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str
    # pass it
    _tts_client: TTSClient
    _file_repository: FileRepository
    tts_sample_rate: int
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, 
                public_id: str, 
                file_repository: FileRepository,
                tts_sample_rate: int, 
                logger, device="cuda", 
                tts_name="xtts", 
                eleven_api_token=""):
        self.public_id = public_id
        self._file_repository = file_repository

       

        self.audio_dubbing_manager = AudioDubbingManager(file_repository, device=device)
        
        self._tts_client = None
        self.eleven_api_token = eleven_api_token
        self.choose_tts_client(tts_name, file_repository, device)
        
        self.tts_sample_rate = self._tts_client.sample_rate

        model_path = os.path.abspath("./voice_conv/OpenVoiceV2")
        self._speaker_conv_client = VoiceToneConverter(ckpt_converter_folder=model_path,
                                                    device=device)

        self.logger = logger
        
              
    def clear_result_video(self, path: str):
        # TODO use file repositoty, for update resulted_video.mp4
        if os.path.exists(path):
            os.remove(path)
         
    def synthesize_segment(self, segment: TranslatedTextedSegment, target_lang: str, vocals_path: str, voice_conv: bool = False):
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        file_path = os.path.join(generated_audio_folder, f"{segment.start}_{segment.end}.wav")
        language = map_language_to_code(target_lang, "whisper")
        
        self._tts_client.generate_audio(
            text=segment.translation, 
            source_audio_file=segment.source_file,
            source_text=segment.text, 
            save_path=file_path,
            language=language
        )
        segment.generated_file = file_path
        if voice_conv:
            self.logger.file_logger.info("Step: voice cloning pipeline")
            
            with self._speaker_conv_client as speaker_conv_client:
                speaker = speaker_conv_client.generate_speaker_embedding(segment.generated_file)
                
                styled_audio_folder = self._file_repository.subdir("styled_audio")
                _, audio_name = os.path.split(segment.source_file)
                audio_save_path = os.path.join(styled_audio_folder, audio_name)
                
                cleaned_audio_path = vocals_path.replace(
                    "vocals", "vocals_enhanced"
                )
                source_spekaer = speaker_conv_client.generate_speaker_embedding(cleaned_audio_path)
                print(segment.generated_file)
                speaker_conv_client.tone_color_converter.convert(
                    audio_src_path=segment.generated_file,
                    src_se=speaker,
                    tgt_se=source_spekaer,
                    output_path=audio_save_path,
                )
                segment.generated_file = audio_save_path
            
            
    
    
    def choose_tts_client(self, name: str, file_repository, device):
        if name == "xtts":
            self._tts_client = XTTSClient(file_repository=file_repository, device=device)
        if name == "fish":
            self._tts_client = FishSpeechClient(file_repository=file_repository, device=device)
        elif name == "elevenlabs":
            self._tts_client = ElevenTTSClient(self.eleven_api_token)
        elif name == "f5tts":
            self._tts_client = FlowClient()
        
    def synthesize(self, video_translation: VideoTranslation, source_lang: str, target_lang: str, voice_conv=False, enhance=False) -> VideoTranslation:

        vocals_audio = video_translation.background_audio["vocals.wav"]
        # self._file_repository.materialize_file(vocals_audio)

        db_manager = AudioDubbingManager(file_repository=self._file_repository)
        AudioDubbingManager.resample_save(vocals_audio, self.tts_sample_rate)
        self.logger.file_logger.info("Resampled vocals audio")
        
        splitted_audio_folder = self._file_repository.subdir("splitted_audio")
        video_translation = db_manager.split_audio_seconds(video_translation,
                                            vocals_audio,
                                            splitted_audio_folder,
                                            sample_rate=self.tts_sample_rate,
                                            )
        self._file_repository.save_dir(self._file_repository.subdir('splitted_audio'))
        
        if enhance:
            self.logger.file_logger.info("Step: resampling pipeline on splitted audio")
            enhanced_audio_folder = self._file_repository.subdir("enhanced_audio")
            video_translation = db_manager.enhance_pipeline(video_translation, enhanced_audio_folder)
            self._file_repository.save_dir(self._file_repository.subdir('enhanced_audio'))
        

        self.logger.file_logger.info("Step: text to speech basic pipeline")
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        video_translation = self._tts_client.tts_pipeline(
                    video_translation,
                    generated_audio_folder,
                    language=target_lang)
        self._file_repository.save_dir(self._file_repository.subdir('generated_audio'))
        
        if voice_conv:
            self.logger.file_logger.info("Step: voice cloning pipeline")
            styled_audio_folder = self._file_repository.subdir("styled_audio")
            with self._speaker_conv_client as speaker_conv_client:
                video_translation = speaker_conv_client.voice_conversion_pipeline(
                    video_translation,
                    styled_audio_folder,
                    source_lang=source_lang
                )
            self._file_repository.save_dir(self._file_repository.subdir('styled_audio'))
        
        new_video_translation = VideoTranslation(
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=video_translation.extracted_audio,
            background_audio=video_translation.background_audio,
            vad_filtered_audio=video_translation.vad_filtered_audio,
            recognized_texts=video_translation.recognized_texts,
            translated_texts=video_translation.translated_texts,
        )

        return new_video_translation
