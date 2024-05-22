import io
import os.path

import torch
import torchaudio

from logging import getLogger


from src.ffmpeg import FFmpegClient
from src.pipeline_models import VideoTranslation
from src.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from src.text_to_speech_service.demucs_client import DemucsClient
from src.text_to_speech_service.tts_client import TTSClient, ElevenLabsTTSClient
from src.utils import upload_file_to_s3, download_from_s3

logger = getLogger(__name__)


class TextToSpeechManager:
    directory: str
    public_id: str

    _tts_client: TTSClient
    sample_rate: int = 16_000
    tts_sample_rate: int = 44_100
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, public_id: str, directory: str = None):
        self.public_id = public_id
        self._tts_client = ElevenLabsTTSClient('f805d6de7a8d5d6f7c0341e62b24b98a')

        if directory is None:
            directory = os.path.join('/Users/nikolaypakhtusov/', 'data', public_id)

        self.directory = directory

        self.audio_dubbing_manager = AudioDubbingManager(self.tts_sample_rate,
                                                         self.directory)

    def synthesize(self, video_translation: VideoTranslation) -> VideoTranslation:
        os.makedirs(self.directory, exist_ok=True)

        vad_filtered_audio_path = os.path.join(self.directory, 'vad_filtered_audio_path')
        download_from_s3(video_translation.vad_filtered_audio_url, vad_filtered_audio_path)

        voice = self._tts_client.clone_voice(vad_filtered_audio_path)
        audios = self._tts_client.generate_audio(video_translation.recognized_texts, voice)

        extracted_audio_file_path = os.path.join(self.directory, 'extracted_audio')
        download_from_s3(video_translation.extracted_audio_url, extracted_audio_file_path)

        video_length = FFmpegClient().get_audio_length(extracted_audio_file_path)

        generated_audio_path = self.audio_dubbing_manager.dub(
            video_translation.recognized_texts,
            audios,
            video_length,
        )

        source_video_path = os.path.join(self.directory, 'source_video')
        download_from_s3(video_translation.extracted_audio_url, source_video_path)

        resulted_audio_path = self._merge_background(
            source_audio_file_path=extracted_audio_file_path,
            voice_audio_path=generated_audio_path)

        resulted_video_path = os.path.join(self.directory, 'resulted_video.mp4')

        FFmpegClient().replace_audio(source_video_path,
                                     resulted_audio_path,
                                     resulted_video_path)

        with open(resulted_video_path, 'rb') as f:
            processed_video_link = upload_file_to_s3(io.BytesIO(f.read()), self.public_id)
        return VideoTranslation(
            source_url=video_translation.source_url,
            extracted_audio_url=video_translation.extracted_audio_url,
            vad_filtered_audio_url=video_translation.vad_filtered_audio_url,
            recognized_texts=video_translation.recognized_texts,
            translated_texts=video_translation.translated_texts,
            processed_video=processed_video_link,
        )

    def _merge_background(self, source_audio_file_path: str, voice_audio_path: str) -> str:
        target_file_path = os.path.join(self.directory, 'demucs_result.wav')
        DemucsClient().separate(source_file_path=source_audio_file_path,
                                target_file_path=target_file_path)

        background_sound, sr_back = torchaudio.load(voice_audio_path)
        speech_audio, sr_speech = torchaudio.load(target_file_path)
        assert sr_back == sr_speech, "Background sr is not equal to speech sr."

        def _fix_tensor_len_by_cutting_to_min(first: torch.Tensor, second: torch.Tensor) \
                -> tuple[torch.Tensor, torch.Tensor]:
            min_length = min(first.shape[-1], second.shape[-1])

            def _slice_multidimensional(tensor: torch.Tensor) -> torch.Tensor:
                split = torch.split(tensor, min_length, dim=(tensor.shape[0] - 1))
                return split[0]

            first = _slice_multidimensional(first)
            second = _slice_multidimensional(second)

            return first, second

        background_sound, speech_audio = _fix_tensor_len_by_cutting_to_min(background_sound, speech_audio)

        common_sound = background_sound + speech_audio

        merged_with_background_audio_path = os.path.join(self.directory, "merged_with_background.wav")
        torchaudio.save(merged_with_background_audio_path, common_sound, sample_rate=self.tts_sample_rate)

        return merged_with_background_audio_path
