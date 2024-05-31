import torch
import torchaudio

from logging import getLogger

from src.ml.api_client import APIClient
from src.pipeline_models.enums import ProcessStatus
from src.ml.ffmpeg import FFmpegClient
from src.file_repository import FileRepository
from src.pipeline_models.models import VideoTranslation
from src.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from src.ml.text_to_speech_service.demucs_client import DemucsClient
from src.ml.text_to_speech_service.tts_client import TTSClient, XTTSClient
from src.ml.speech_to_text_service import VadClient
from pyrubberband.pyrb import time_stretch

logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str

    _tts_client: TTSClient
    _api_client: APIClient
    _file_repository: FileRepository
    sample_rate: int = 24_000
    sample_rate: int = 24_000
    tts_sample_rate: int = 24_000
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository):
        self.public_id = public_id
        self._api_client = api_client
        self._file_repository = file_repository

        self.audio_dubbing_manager = AudioDubbingManager(self.tts_sample_rate,
                                                         file_repository)
        self._tts_client = XTTSClient()
    
    def get_audio_length(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        return audio.shape[1] / sr

    def synthesize(self, video_translation: VideoTranslation) -> VideoTranslation:

        vocals_audio = video_translation.background_audio["vocals.wav"]
        self._file_repository.materialize_file(vocals_audio)

        db_manager = AudioDubbingManager(file_repository=self._file_repository,
                                         tts_sample_rate=self.tts_sample_rate)
        
        AudioDubbingManager.resample_save(vocals_audio.file_path,
                        target_sr=self.tts_sample_rate)
        
        # split source -> generate tts -> style from tts
        df = db_manager.split_audio_seconds(video_translation.recognized_texts,
                                            vocals_audio.file_path,
                                            sample_rate=self.tts_sample_rate)
        
        df_generated_audio = self._tts_client.generate_audio(
                    video_translation.translated_texts,
                    vocals_audio,
                    df)
        df_styled_audio = self._tts_client.style_audio(
                    df_generated_audio)

        extracted_audio_file = self._file_repository.materialize_file(
            video_translation.extracted_audio
        )
        video_length = FFmpegClient().get_audio_length(extracted_audio_file.file_path)

        generated_audio = self.merge_timestamps_speedup(
            df_styled_audio,
            video_length=video_length,
            source_sample_rate=self.sample_rate
        )

        # TODO: save correctly if need on the s3
        styled_audio = self._file_repository.get_file("styled_full_audio.wav")
        torchaudio.save(styled_audio.file_path, generated_audio, self.sample_rate)

        audio_backgrounds = {
            name: self._file_repository.materialize_file(remote_file).file_path
            for name, remote_file in
            video_translation.background_audio.items()
        }

        resulted_audio = DemucsClient().merge_background(
                    styled_audio.file_path,
                    audio_backgrounds,
        )
        
        result_audio = self._file_repository.get_file("resulted_audio.wav")
        torchaudio.save(result_audio.file_path, resulted_audio, self.sample_rate)

        source_video = self._file_repository.materialize_file(
            video_translation.source_file
        )
        resulted_video = self._file_repository.get_file('resulted_video.mp4')

        # TODO: it will not work without not saved properly resulted video
        FFmpegClient().replace_audio(source_video.file_path,
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
            processed_video=resulted_video,
        )

        self._api_client.update_video(self.public_id,
                                      new_video_translation,
                                      progress=10,
                                      status=ProcessStatus.done)

        return new_video_translation

    def merge_timestamps_speedup(self, df, video_length, source_sample_rate):
        """
        Algorithm that work on time stretching - not the best one.
        """
        df['gen_dur'] = df['styled_generated_path'].apply(lambda audio_path: self.get_audio_length(audio_path))
        df['pause'] = df['start'].shift(-1) - df['end'] # пауза между двумя предложениями 
        df['pause'] = df['pause'].fillna(0)
        df['dur_gen_pause'] = df['gen_dur'] + df['pause'] # длина сгенерированной речи + пауза, которую можно сделать 
        df['place_gen'] = df['end'] - df['start'] + df['pause'] # место, которое можно поставить для сгенерированной фразы 
        df['need_time'] = df['gen_dur'] - df['place_gen'] # сколько времени необходимо, если < 0 - то, все ок, если > 0, то нужно что-то сделать  
        df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        df['need_speedup'] = df['gen_dur'] > df['place_gen']
        df['duration_orig'] = df['end'] - df['start']
        df['speed_rate'] = df['dur_gen_pause'] / df['duration_orig']
        full_audio_blank = torch.zeros((1, int(video_length * source_sample_rate)))

        for i, line in df.iterrows():
            audio, sr = torchaudio.load(line.styled_generated_path)

            start = line.start
                    
            audio = time_stretch(audio.squeeze().numpy(), sr, rate=line.speed_rate)
            audio = torch.tensor(audio).unsqueeze(0)

            start_pos = int(start*source_sample_rate)
            end_pos = int(start_pos + audio.shape[-1])
            
            full_audio_blank[0, int(start_pos): int(end_pos)] = audio[0]
        return full_audio_blank

    def merge_timestamps_stretch_whole(self, df):
        previous_pause = torch.zeros((1, int(df.iloc[0].start * self.sample_rate)))
        audio_first, sr = torchaudio.load(df.iloc[0].styled_generated_path)
        pause = torch.zeros((1, int(df.iloc[0].pause * self.sample_rate)))

        audio_first = torch.cat((previous_pause, audio_first, pause), dim=1)

        for i, line in df.iterrows():
            audio, sr = torchaudio.load(line.styled_generated_path)
            pause = torch.zeros((1, int(line.pause * self.sample_rate)))

            if i == 0:
                continue

            audio_first = torch.cat((audio_first, audio), dim=1)
            audio_first = torch.cat((audio_first, pause), dim=1)
        # add end pause
        return audio_first
