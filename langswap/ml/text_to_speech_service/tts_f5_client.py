import sys

# TODO: change the paths
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../F5-TTS/src"))
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

from f5_tts.model import DiT
from cached_path import cached_path
import re
import tomli
import soundfile as sf
import os
from tqdm import tqdm
import numpy as np

import torchaudio
from ruaccent import RUAccent


class FlowClient:
    def __init__(
        self,
        vocab_file='./models_weights/ESpeech-TTS/vocab.txt',
        vocos_local_path: str = "./models_weights/vocos-mel-24khz",
        model_path: str = "./models_weights/ESpeech-TTS/model_rlv2.pt",
    ):
        self.vocab_file = vocab_file
        self.sample_rate = 24000
        self.model_path = model_path

        # where do we use vocos?
        self.vocos = load_vocoder(is_local=True, local_path=vocos_local_path)
        
        self.accentizer = RUAccent()
        self.tts = self.load_tts_flow()

    def load_tts_flow(self):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        self.accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False, workdir="./models_weights/ruaccent")
        model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg,
            ckpt_path=self.model_path,
            mel_spec_type="vocos",
            vocab_file=self.vocab_file,
            device="cuda")
        return model

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration=None
    ):
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text)
        reg2 = r"\[(\w+)\]"

        for text in chunks:
            match = re.match(reg2, text)
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            if language == "russian":
                gen_text = self.accentizer.process_all(gen_text)
            
            print(f"Duration within generate_audio: {duration}")
            source_audio_file, source_text = preprocess_ref_audio_text(
                source_audio_file, 
                source_text, 
            )
                    
            audio, final_sample_rate, _ = infer_process(
                ref_audio=source_audio_file,
                ref_text=source_text,
                gen_text=gen_text,
                model_obj= self.tts,
                vocoder=self.vocos,
                mel_spec_type="vocos",
                speed=1.5,
                # fix_duration=duration+7 # 7 is a magic number 
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
                duration = (segment.end - segment.start)
                print(f"Duration: {duration}")
                self.generate_audio(
                    segment.translation, segment.source_file, segment.text, file_path, language, duration=duration
                )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
