import torch
import torchaudio

from logging import getLogger

from src.api_client import APIClient
from src.enums import ProcessStatus
from src.ffmpeg import FFmpegClient
from src.file_repository import FileRepository, RemoteFile
from src.pipeline_models import VideoTranslation
from src.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from src.text_to_speech_service.demucs_client import DemucsClient
from src.text_to_speech_service.tts_client import TTSClient, ElevenLabsTTSClient

logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str

    _tts_client: TTSClient
    _api_client: APIClient
    _file_repository: FileRepository
    sample_rate: int = 16_000
    tts_sample_rate: int = 44_100
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository):
        self.public_id = public_id
        self._tts_client = ElevenLabsTTSClient('f805d6de7a8d5d6f7c0341e62b24b98a')
        self._api_client = api_client
        self._file_repository = file_repository

        self.audio_dubbing_manager = AudioDubbingManager(self.tts_sample_rate,
                                                         file_repository)

    def synthesize(self, video_translation: VideoTranslation) -> VideoTranslation:

        vad_filtered_audio_file = self._file_repository.materialize_file(
            video_translation.vad_filtered_audio
        )

        voice = self._tts_client.clone_voice(vad_filtered_audio_file.file_path)
        audios = self._tts_client.generate_audio(video_translation.recognized_texts, voice)

        extracted_audio_file = self._file_repository.materialize_file(
            video_translation.extracted_audio
        )
        video_length = FFmpegClient().get_audio_length(extracted_audio_file.file_path)

        generated_audio = self.audio_dubbing_manager.dub(
            video_translation.recognized_texts,
            audios,
            video_length,
        )

        source_video = self._file_repository.materialize_file(
            video_translation.source_file
        )

        resulted_audio = self._merge_background(
            source_audio_file_path=extracted_audio_file.file_path,
            voice_audio_path=generated_audio.file_path)

        resulted_video = self._file_repository.get_file('resulted_video.mp4')

        FFmpegClient().replace_audio(source_video.file_path,
                                     resulted_audio.file_path,
                                     resulted_video.file_path)
        self._file_repository.save_file(resulted_video)

        new_video_translation = VideoTranslation(
            source_file=video_translation.source_file,
            extracted_audio=video_translation.extracted_audio,
            vad_filtered_audio=video_translation.vad_filtered_audio,
            recognized_texts=video_translation.recognized_texts,
            translated_texts=video_translation.translated_texts,
            processed_video=resulted_video,
        )

        self._api_client.update_video(self.public_id,
                                      new_video_translation,
                                      progress=10,
                                      status=ProcessStatus.done)

        return new_video_translation

    def _merge_background(self, source_audio_file_path: str, voice_audio_path: str) -> RemoteFile:
        demucs_result_file = self._file_repository.get_file('demucs_result.wav')
        DemucsClient().separate(source_file_path=source_audio_file_path,
                                target_file_path=demucs_result_file.file_path)

        background_sound, sr_back = torchaudio.load(voice_audio_path)
        speech_audio, sr_speech = torchaudio.load(demucs_result_file.file_path)
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

        merged_with_background_audio = self._file_repository.get_file("merged_with_background.wav")
        torchaudio.save(merged_with_background_audio.file_path, common_sound, sample_rate=self.tts_sample_rate)

        return merged_with_background_audio
