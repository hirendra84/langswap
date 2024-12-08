import torchaudio
import os

from logging import getLogger

from src.ml.api_client import APIClient
from src.pipeline_models.enums import ProcessStatus
from src.ml.ffmpeg import FFmpegClient
from src.file_repository import FileRepository
from src.pipeline_models.models import VideoTranslation
from src.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from src.ml.text_to_speech_service.video_dubbing_manager import VideoDubbingManager
from src.ml.text_to_speech_service.demucs_client import DemucsClient
from src.ml.text_to_speech_service.tts_xtts_client import XTTSClient
from src.ml.text_to_speech_service.tts_f5_client import FlowClient
from src.ml.text_to_speech_service.tts_eleven_client import ElevenTTSClient
from src.ml.text_to_speech_service.voice_converter import VoiceToneConverter

logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str
    # pass it
    _tts_client: XTTSClient
    _api_client: APIClient
    _file_repository: FileRepository
    tts_sample_rate: int = 24_000
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository,
                tts_sample_rate: int, logger, device="cuda", tts_name="xtts"):
        self.public_id = public_id
        self._api_client = api_client
        self._file_repository = file_repository

        self.tts_sample_rate = tts_sample_rate

        self.audio_dubbing_manager = AudioDubbingManager(file_repository)
        self.video_dubbing_manager = VideoDubbingManager(file_repository, logger)
        self._tts_client = None
        self.choose_tts_client(tts_name, file_repository, device)

        model_path = os.path.abspath("./voice_conv/OpenVoiceV2")
        self._speaker_conv_client = VoiceToneConverter(ckpt_converter_folder=model_path,
                                                    device=device)

        self.logger = logger
        self.audio_extensions = ["mp3", "wav", "MP3"]
    
    def choose_tts_client(self, name: str, file_repository, device):
        if name == "xtts":
            self._tts_client = XTTSClient(file_repository=file_repository, device=device)
        elif name == "elevenlabs":
            self._tts_client = ElevenTTSClient()
        elif name == "f5tts":
            self._tts_client = FlowClient()
        
    def synthesize(self, video_translation: VideoTranslation, source_lang: str, target_lang: str, voice_conv=False, enhance=False, merge_pipeline="pause_based") -> VideoTranslation:

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
        
        if enhance:
            self.logger.file_logger.info("Step: resampling pipeline on splitted audio")
            enhanced_audio_folder = self._file_repository.subdir("enhanced_audio")
            video_translation = db_manager.enhance_pipeline(video_translation, enhanced_audio_folder)
        
        self.logger.file_logger.info("Step: text to speech basic pipeline")
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        video_translation = self._tts_client.tts_pipeline(
                    video_translation,
                    generated_audio_folder,
                    language=target_lang)
        
        if voice_conv:
            self.logger.file_logger.info("Step: voice cloning pipeline")
            styled_audio_folder = self._file_repository.subdir("styled_audio")
            video_translation = self._speaker_conv_client.voice_conversion_pipeline(
                video_translation,
                styled_audio_folder,
                source_lang=source_lang
            )
        
        if merge_pipeline == "pause_based":
            generated_audio, generated_sr = self.video_dubbing_manager.merge_timestamps_pause_based(
                video_translation,
                vocals_audio
            )
        elif merge_pipeline == "stretch_whole":
            generated_audio, generated_sr = self.video_dubbing_manager.merge_timestamps_stretch_whole(
                video_translation,
                vocals_audio
            ) 
        elif merge_pipeline == "speedup":
            generated_audio, generated_sr = self.video_dubbing_manager.merge_timestamps_speedup(
                video_translation,
                vocals_audio
            )

        # TODO: save correctly if need on the s3
        styled_audio = self._file_repository.get_file("styled_full_audio.wav")
        torchaudio.save(styled_audio.file_path, generated_audio, generated_sr)

        # audio_backgrounds = {
        #     name: self._file_repository.materialize_file(remote_file).file_path
        #     for name, remote_file in
        #     video_translation.background_audio.items()
        # }
        audio_backgrounds = {
            name: remote_file
            for name, remote_file in
            video_translation.background_audio.items()
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
        source_video = video_translation.source_file.file_path

        base, extension = os.path.splitext(video_translation.source_file.file_path)

        if extension not in self.audio_extensions:
            FFmpegClient().replace_audio(source_video,
                                        result_audio.file_path,
                                        resulted_video.file_path)
            self._file_repository.save_file(resulted_video)

        new_video_translation = VideoTranslation(
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=video_translation.extracted_audio,
            vad_filtered_audio=video_translation.vad_filtered_audio,
            recognized_texts=video_translation.recognized_texts,
            translated_texts=video_translation.translated_texts,
            processed_video=resulted_video
        )

        self._api_client.update_video(self.public_id,
                                      new_video_translation,
                                      progress=10,
                                      status=ProcessStatus.done)
        return new_video_translation
