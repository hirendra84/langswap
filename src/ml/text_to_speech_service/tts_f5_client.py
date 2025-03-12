import sys

# TODO: change the paths
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../F5-TTS"))
from model.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    remove_silence_for_generated_wav,
)

from model import DiT
from cached_path import cached_path
import re
import tomli
import soundfile as sf
import os
from tqdm import tqdm
import numpy as np

import torchaudio


class FlowClient:
    def __init__(
        self,
        config_path="/app/F5-TTS/inference-cli.toml",
        vocab_file='/app/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt',
        vocos_local_path: str = "/app/vocos-mel-24khz",
    ):
        # TODO: change all the paths to relative
        self.config = tomli.load(open(config_path, "rb"))
        self.vocab_file = vocab_file

        # where do we use vocos?
        vocos = load_vocoder(is_local=True, local_path=vocos_local_path)
        self.tts = self.load_tts_flow()

        self.ref_audio = "/app/F5-TTS/test_en_1_ref_short.wav"
        self.ref_text = self.config["ref_text"]

    def load_tts_flow(self):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        ckpt_file = self.config.get("ckpt_file", "")
        if not ckpt_file:
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(
                cached_path(
                    f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"
                )
            )

        model = load_model(model_cls, model_cfg, ckpt_file, self.vocab_file)
        return model

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        save_path: str,
        remove_silence: bool = True,
        duration=None
    ):
        # ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text) # TODO remove and use only the main voice
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text)
        reg2 = r"\[(\w+)\]"

        for text in chunks:
            match = re.match(reg2, text)
            text = re.sub(reg2, "", text)
            gen_text = text.strip()

            audio, final_sample_rate, spectragram = infer_process(
                self.ref_audio, self.ref_text, gen_text, self.tts, fix_duration=duration
            )
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            with open(save_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice generation pipeline.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")

            if not os.path.exists(file_path):
                self.generate_audio(
                    segment.translation, segment.source_file, file_path, language
                )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
