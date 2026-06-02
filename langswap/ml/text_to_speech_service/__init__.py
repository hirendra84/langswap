import torchaudio
import os

from logging import getLogger

from langswap.file_repository import FileRepository
from langswap.pipeline_models.models import VideoTranslation
from langswap.pipeline_models.models import TranslatedTextedSegment
from langswap.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from langswap.ml.text_to_speech_service.tts_client import TTSClient
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code


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
            language=language,
            duration=segment.end - segment.start
        )
        segment.generated_file = file_path

    
    def choose_tts_client(self, name: str, file_repository, device):
        if name == "xtts":
            # Lazy import: XTTS has heavy/optional deps.
            from langswap.ml.text_to_speech_service.tts_xtts_client import XTTSClient

            self._tts_client = XTTSClient(file_repository=file_repository, device=device)
        elif name == "elevenlabs":
            from langswap.ml.text_to_speech_service.tts_eleven_client import ElevenTTSClient

            self._tts_client = ElevenTTSClient(self.eleven_api_token)
        elif name == "f5tts":
            from langswap.ml.text_to_speech_service.tts_f5_client import FlowClient

            self._tts_client = FlowClient()
        elif name == "chatterbox":
            from langswap.ml.text_to_speech_service.tts_chatterbox_client import ChatterboxClient

            self._tts_client = ChatterboxClient(file_repository=file_repository, device=device)
        elif name == "qwen3":
            # Lazy import: qwen-tts is optional.
            from langswap.ml.text_to_speech_service.tts_qwen3_client import Qwen3TTSClient

            self._tts_client = Qwen3TTSClient(device=device)
        elif name == "omnivoice":
            from langswap.ml.text_to_speech_service.tts_omnivoice_client import OmniVoiceClient
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "omnivoice", device),
                lambda: OmniVoiceClient(device=device))
        else:
            raise ValueError(f"Unknown TTS engine: {name}")


    def synthesize(self, video_translation: VideoTranslation, source_lang: str, target_lang: str, voice_conv=False, enhance=False) -> VideoTranslation:

        vocals_audio = video_translation.background_audio["vocals.wav"]

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
