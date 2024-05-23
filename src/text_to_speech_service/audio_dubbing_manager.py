import os
from typing import Iterator

import elevenlabs
import numpy as np
import torchaudio

from tqdm import tqdm

import pandas as pd
import torch

from src.pipeline_models import TextedSegment


class AudioDubbingManager:
    tts_sample_rate: int

    def __init__(self, tts_sample_rate: int, directory: str):
        self.tts_sample_rate = tts_sample_rate
        self.directory = directory

    def dub(self, recognized_texts: list[TextedSegment],
            audios: list[Iterator[bytes]],
            video_length: float) -> str:
        df = self._map_audios(recognized_texts, audios)

        _, generated_audio_path = self._merge_audio_timestamps(df, video_length, self.tts_sample_rate)
        df.to_csv(os.path.join(self.directory, 'audio_frames.csv'))

        return generated_audio_path

    def _map_audios(self, segments: list[TextedSegment], audios: list[Iterator[bytes]]) -> pd.DataFrame:
        # | | |
        df = pd.DataFrame(
            [{
                'text': t.text,
                'start': t.start,
                'end': t.end,
            } for t in segments]
        )
        for i, audio in enumerate(audios):
            generated_audio_path = os.path.join(self.directory, f"generated_audio_pt{i}.wav")
            elevenlabs.save(audio, generated_audio_path)
            df.loc[i, 'syn_audio_path'] = generated_audio_path

        # df['gen_dur'] = df['syn_audio_path'].apply(lambda x: FFmpegClient().get_audio_length(x))
        df['gen_dur'] = df['syn_audio_path'].apply(lambda x: torchaudio.load(x)[0].shape[1] / self.tts_sample_rate)

        df['pause'] = df['start'].shift(-1) - df['end']
        df['dur_gen_pause'] = df['gen_dur'] + df['pause']
        df['place_gen'] = df['end'] - df['start'] + df['pause']
        df['gen_end'] = df['start'] + df['gen_dur']
        df['can_start'] = [0] + df['gen_end'].to_list()[:-1]
        df['need_time'] = df['gen_dur'] - df['place_gen']
        df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        df['need_speedup'] = df['gen_dur'] > df['place_gen']
        df['duration_orig'] = df['end'] - df['start']

        return df

    def _merge_audio_timestamps(self, df: pd.DataFrame, video_length: float, sample_rate: int)\
            -> tuple[torch.Tensor, str]:
        concated_audio_tensor = torch.zeros((1, int(video_length * sample_rate * 1.1)))

        for i, line in tqdm(df.iterrows(), total=df.shape[0]):
            wav, sr = torchaudio.load(line.syn_audio_path)

            start_pos = line.start * sample_rate

            if start_pos < 0:
                start_pos = 0
            # end_pos = start_pos + (FFmpegClient().get_audio_length(line.syn_audio_path) * sample_rate)
            start_pos = int(np.ceil(start_pos))
            end_pos = start_pos + wav.shape[-1]

            print(f'try_num, {i}')
            try:
                concated_audio_tensor[0, start_pos: int(end_pos)] = wav[0]
            except RuntimeError:
                print()
            # torch.frombuffer(bytes, dtype=torch.int32)

        generated_audio_path = os.path.join(self.directory, "merged_audio.wav")
        torchaudio.save(generated_audio_path, concated_audio_tensor, sample_rate=sample_rate)
        print(f" Merged audio shape is {concated_audio_tensor.shape}.")
        return concated_audio_tensor, generated_audio_path
