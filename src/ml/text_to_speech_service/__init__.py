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
from src.ml.text_to_speech_service.tts_client import TTSClient, XTTSClient, VoiceToneConverter
from src.ml.speech_to_text_service import VadClient
from pyrubberband.pyrb import time_stretch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str

    _tts_client: XTTSClient
    _api_client: APIClient
    _file_repository: FileRepository
    tts_sample_rate: int = 24_000
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository, tts_sample_rate: int, device="cuda"):
        self.public_id = public_id
        self._api_client = api_client
        self._file_repository = file_repository

        self.tts_sample_rate = tts_sample_rate

        self.audio_dubbing_manager = AudioDubbingManager(file_repository)
        self._tts_client = XTTSClient(file_repository=file_repository, device=device)
        self._speaker_conv_client = VoiceToneConverter(ckpt_converter_folder="/home/milana/OpenVoice/OpenVoiceV2/",
                                                    device=device)

    def synthesize(self, video_translation: VideoTranslation, source_lang: str, voice_conv=False, enhance=False, merge_pipeline="pause_based") -> VideoTranslation:

        vocals_audio = video_translation.background_audio["vocals.wav"]
        # self._file_repository.materialize_file(vocals_audio)

        db_manager = AudioDubbingManager(file_repository=self._file_repository)
        
        AudioDubbingManager.resample_save(vocals_audio.file_path, self.tts_sample_rate)
        
        splitted_audio_folder = self._file_repository.subdir("splitted_audio")
        video_translation = db_manager.split_audio_seconds(video_translation,
                                            vocals_audio.file_path,
                                            splitted_audio_folder,
                                            sample_rate=self.tts_sample_rate,
                                            )
        
        if enhance:
            enhanced_audio_folder = self._file_repository.subdir("enhanced_audio")
            video_translation = db_manager.enhance_pipeline(video_translation, enhanced_audio_folder)
            
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        video_translation = self._tts_client.tts_pipeline(
                    video_translation,
                    generated_audio_folder)
        
        if voice_conv:
            styled_audio_folder = self._file_repository.subdir("styled_audio")
            video_translation = self._speaker_conv_client.voice_conversion_pipeline(
                video_translation,
                styled_audio_folder,
                source_lang=source_lang
            )
        
        if merge_pipeline == "pause_based":
            generated_audio, generated_sr = self.merge_timestamps_pause_based(
                video_translation,
                vocals_audio
            )
        elif merge_pipeline == "stretch_whole":
            generated_audio, generated_sr = self.merge_timestamps_stretch_whole(
                video_translation,
                vocals_audio
            ) 

        # TODO: save correctly if need on the s3
        styled_audio = self._file_repository.get_file("styled_full_audio.wav")
        torchaudio.save(styled_audio.file_path, generated_audio, generated_sr)

        audio_backgrounds = {
            name: self._file_repository.materialize_file(remote_file).file_path
            for name, remote_file in
            video_translation.background_audio.items()
        }

        resulted_audio, save_sr = DemucsClient().merge_background(
                    styled_audio.file_path,
                    audio_backgrounds,
        )
        
        result_audio = self._file_repository.get_file("resulted_audio.wav")
        torchaudio.save(result_audio.file_path, resulted_audio, save_sr)

        resulted_video = self._file_repository.get_file("resulted_video.mp4")
        source_video = video_translation.source_file.file_path

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
            processed_video=resulted_video,
        )

        self._api_client.update_video(self.public_id,
                                      new_video_translation,
                                      progress=10,
                                      status=ProcessStatus.done)
        return new_video_translation
    
    def merge_timestamps_pause_based(self, video_translation, vocals_audio):
        """
        Algorithm that works on filling the pauses - not the best one.
        """
        df = pd.DataFrame()
        df["start"] = [segment.start for segment in video_translation.translated_texts] 
        df["end"] = [segment.end for segment in video_translation.translated_texts] 
        df["pause"] = df["start"].shift(-1) - df["end"]
        df['gen_dur'] = [torchaudio.load(segment.generated_file)[0].shape[1] / 24000 for segment in video_translation.translated_texts]
        df['pause'] = df['start'].shift(-1) - df['end']
        df['dur_gen_pause'] = df['gen_dur'] + df['pause']
        df['place_gen'] = df['end'] - df['start'] + df['pause']
        df['gen_end'] = df['start'] + df['gen_dur']
        df['can_start'] = [0] + df['gen_end'].to_list()[:-1]
        df['need_time'] = df['gen_dur'] - df['place_gen']
        df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        df['need_speedup'] = df['gen_dur'] > df['place_gen']
        df['duration_orig'] = df['end'] - df['start']

        prev_audio, sr = torchaudio.load(vocals_audio.file_path)
        prev_audio_shape = prev_audio.shape[1]        
        blank_audio_tensor = torch.zeros((1, int(prev_audio_shape)))

        for idx, segment in enumerate(video_translation.translated_texts):
            audio, sr = torchaudio.load(segment.generated_file)
            start_pos = df.loc[idx, 'new_start'] * sr

            start_pos = np.ceil(start_pos)
            end_pos = start_pos + audio.shape[-1]
            end_pos = np.ceil(end_pos)

            blank_audio_tensor[0, int(start_pos): int(end_pos)] = audio[0]

        generated_audio_path = "merged_audio_algo.wav"
        torchaudio.save(generated_audio_path, blank_audio_tensor, sample_rate=sr)
        return blank_audio_tensor, sr

    def merge_timestamps_speedup(self, df, video_length, source_sample_rate):
        """
        Algorithm that works on time stretching - not the best one.
        """
        df['gen_dur'] = df['styled_generated_path'].apply(lambda audio_path: self.get_audio_length(audio_path))
        df['pause'] = df['start'].shift(-1) - df['end'] # пауза между двумя предложениями 
        df['pause'] = df['pause'].fillna(0)
        df['dur_gen_pause'] = df['gen_dur'] + df['pause'] # длина сгенерированной речи + пауза, которую можно сделать 
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

    def merge_timestamps_stretch_whole(self, video_translation, vocals_audio):
        prev_audio, sr = torchaudio.load(vocals_audio.file_path)
        prev_audio_shape = prev_audio.shape[1]
        target_audio_length = prev_audio_shape / sr

        df = pd.DataFrame()
        df["start"] = [segment.start for segment in video_translation.translated_texts] 
        df["end"] = [segment.end for segment in video_translation.translated_texts] 
        df["pause"] = df["start"].shift(-1) - df["end"] # пауза между двумя предложениями 
        if df.shape[0] > 1:
            df.loc[df.shape[0] - 1, "pause"] = target_audio_length - df.loc[df.shape[0] - 1, "end"] # the last pause
        elif df.shape[0] == 1:
            df.loc[0, "pause"] = 0

        audio_first, sr_generated = torchaudio.load(video_translation.translated_texts[0].generated_file)
        previous_pause = torch.zeros((1, int(video_translation.translated_texts[0].start * sr_generated)))
        pause = torch.zeros((1, int(df.iloc[0].pause * sr_generated)))

        audio_generated = torch.cat((previous_pause, audio_first, pause), dim=1)

        for idx, segment in enumerate(video_translation.translated_texts):
            if idx == 0:
                continue
            audio, sr = torchaudio.load(segment.generated_file)
            pause = torch.zeros((1, int(df.loc[idx, "pause"] * sr_generated)))

            audio_generated = torch.cat((audio_generated, audio), dim=1)
            audio_generated = torch.cat((audio_generated, pause), dim=1)

        generated_audio_length = audio_generated.shape[1] / sr_generated

        stretched = time_stretch(audio_generated.squeeze().numpy(), sr=sr_generated, rate=generated_audio_length/target_audio_length)
        stretched = torch.tensor(stretched).unsqueeze(0)
        return stretched, sr_generated
