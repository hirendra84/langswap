from abc import ABC
from typing import Iterator

import elevenlabs
from elevenlabs.client import ElevenLabs
from tqdm.auto import tqdm

from src.pipeline_models import TextedSegment


class TTSClient(ABC):

    def __init__(self, api_key: str):
        ...

    def clone_voice(self, voice_path: str, voice_descr: str = '', voice_name = '' ):
        ...

    def generate_audio(self, data: list[TextedSegment], voice: elevenlabs.Voice) -> list[Iterator[bytes]]:
        ...


class ElevenLabsTTSClient(TTSClient):

    _client: ElevenLabs

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = ElevenLabs(api_key=api_key)

    def clone_voice(self,
                    voice_path: str,
                    voice_descr: str = 'Default description',
                    voice_name: str = 'Default voice') -> elevenlabs.Voice:
        exception = None
        for _ in range(10):
            try:
                voice = self._client.clone(voice_name, description=voice_descr,
                                           files=[voice_path])
                return voice
            except elevenlabs.core.api_error.ApiError as e:
                exception = e
                self._remove_first_voice()
        else:
            raise exception

    def _remove_first_voice(self):
        voices = self._client.voices.get_all().voices
        for voice in voices:
            if voice.category != 'cloned':
                continue
            voice_id = voice.voice_id
            self._client.voices.delete(voice_id)
            return

    def generate_audio(self, data: list[TextedSegment], voice: elevenlabs.Voice) -> list[Iterator[bytes]]:
        audios: list[Iterator[bytes]] = []
        for i, segment in enumerate(data):
            audio: Iterator[bytes] = self._client.generate(text=segment.text, voice=voice)
            audios.append(audio)

        return audios

        #     generated_audio_path = os.path.join(self.cfg.temp_dir, f"{i}.wav")
        #     save(audio, generated_audio_path)
        #     df.loc[i, 'syn_audio_path'] = generated_audio_path
        #
        # df['gen_dur'] = df['syn_audio_path'].apply(lambda x: torchaudio.load(x)[0].shape[1] / self.cfg.tts_sample_rate)
        # df['pause'] = df['start'].shift(-1) - df['end']
        # df['dur_gen_pause'] = df['gen_dur'] + df['pause']
        # df['place_gen'] = df['end'] - df['start'] + df['pause']
        # df['gen_end'] = df['start'] + df['gen_dur']
        # df['can_start'] = [0] + df['gen_end'].to_list()[:-1]
        # df['need_time'] = df['gen_dur'] - df['place_gen']
        # df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        # df['need_speedup'] = df['gen_dur'] > df['place_gen']
        # df['duration_orig'] = df['end'] - df['start']
        # return df



