import torch


class VadClient:

    def __init__(self):

        self.model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False,
                                               onnx=False)
        (self.vad_get_speech_timestamps, self.vad_save_audio, self.vad_read_audio, self.VADIterator,
         self.vad_collect_chunks) = utils

    def vad_filter(self, source_audio_path: str, target_audio_path: str, sample_rate) -> str:
        wav = self.vad_read_audio(source_audio_path, sampling_rate=sample_rate)
        speech_timestamps = self.vad_get_speech_timestamps(wav, self.model_vad, sampling_rate=sample_rate,
                                                           return_seconds=False)
        target_audio_path = f'{target_audio_path}.wav'
        self.vad_save_audio(target_audio_path, self.vad_collect_chunks(speech_timestamps, wav),
                            sampling_rate=sample_rate)
        return target_audio_path
