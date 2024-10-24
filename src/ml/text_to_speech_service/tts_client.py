from abc import ABC
import re
from tqdm.auto import tqdm
import tomli
import numpy as np
from src.pipeline_models.models import TranslatedTextedSegment

import sys
sys.path.append("/app/coqui")

import soundfile as sf

from coqui.TTS.api import TTS
import os
from src.file_repository import FileRepository
from src.utils.ml_processing.lang2code_mapper import map_language_to_code

import torchaudio
import os
import torch

import sys
sys.path.append("/app/voice_conv")

from voice_conv.openvoice import se_extractor
from voice_conv.openvoice.api import ToneColorConverter

from pydub import AudioSegment

import sys
sys.path.append("/home/milana/app/local_check_data/app/F5-TTS")

from model.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

from model import DiT, UNetT
from cached_path import cached_path



def add_pauses(audio_path: str, num_sec=2):
    audio, sr = torchaudio.load(audio_path)

    pause_start = torch.zeros((1, sr*num_sec))
    pause_end = torch.zeros((1, sr*num_sec))

    audio = torch.cat([pause_start, audio, pause_end], dim=1)

    torchaudio.save(audio_path, audio, sr)


class TTSClient(ABC):

    def __init__(self):
        ...

    def clone_voice(self, voice_path: str, voice_descr: str = '', voice_name = '' ):
        ...

    def generate_audio(self, text: str, source_audio_file: str, save_path: str):
        ...
        
    def tts_pipelene(self, data: list[TranslatedTextedSegment], output_folder: str) -> list[TranslatedTextedSegment]:
        ...


class VoiceToneConverter:
    def __init__(self,
                ckpt_converter_folder: str,
                device="cpu"):
        self.ckpt_converter_folder = ckpt_converter_folder
        config_path = os.path.join(self.ckpt_converter_folder, "converter/config.json")
        checkpoint_path = os.path.join(self.ckpt_converter_folder, "converter/checkpoint.pth")

        self.tone_color_converter = ToneColorConverter(config_path, device=device)
        self.tone_color_converter.load_ckpt(checkpoint_path)

    def create_speaker(self, audio_path: str):
        se, _ = se_extractor.get_se(audio_path, self.tone_color_converter, vad=True)
        return se

    def merge_enhanced(self, video_translation):
        combined_audio = AudioSegment.empty()

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Merge enhanced pipeline.', leave=True)):
            audio_segment = AudioSegment.from_file(segment.source_file)
            combined_audio += audio_segment

            save_path = video_translation.background_audio["vocals.wav"]
            combined_audio.export(save_path.replace("vocals", "vocals_enhanced"), format="wav")

    def voice_conversion_pipeline(self,
                            video_translation,
                            temp_folder,
                            source_lang):
        self.merge_enhanced(video_translation)
        clean_audio_speaker = video_translation.background_audio["vocals.wav"].replace("vocals", "vocals_enhanced")
        clean_audio_speaker = self.create_speaker(clean_audio_speaker)

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Voice conversion pipeline.', leave=True)):            
            folder_path, audio_name = os.path.split(segment.source_file)
            audio_save_path = os.path.join(temp_folder, audio_name)

            speaker = self.create_speaker(segment.generated_file)
            
            self.tone_color_converter.convert(
                audio_src_path=segment.generated_file,
                src_se=speaker,
                tgt_se=clean_audio_speaker,
                output_path=audio_save_path
            )
        
            video_translation.translated_texts[idx].generated_file = audio_save_path
        return video_translation


class XTTSClient:
    def __init__(self,
                file_repository: FileRepository,
                tts_model_path="./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2",
                tts_model_id="tts_models/multilingual/multi-dataset/xtts_v2",
                device="cuda"):
        gpu = True if device == "cuda" else False
        tts_model_path = os.path.abspath(tts_model_path)

        # config_path = "./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2/config.json"
        # config_path = os.path.abspath(config_path)

        config_path = os.path.join(tts_model_path, "config.json")

        self.tts = TTS(model_path=tts_model_path, config_path=config_path)
        self._file_repository = file_repository
    
    def generate_audio(self, text: str, source_audio_path: str, save_path: str, language: str):
        """
        Generates and styles audio, saves according to the path.
        : param style: whether voice conversion should be applied
        """
        self.tts.tts_to_file(text=text, file_path=save_path, speaker_wav=source_audio_path, language=language, enable_text_splitting=False, repetition_penalty=2.0)

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        language = map_language_to_code(language, "whisper")

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Voice generation pipeline.', leave=True)):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")            

            if not os.path.exists(file_path):
                add_pauses(segment.source_file)
                self.generate_audio(
                                    segment.translation,
                                    segment.source_file,
                                    file_path,
                                    language
                                    )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
    
class FlowClient:
    def __init__(self, config_path: str, vocab_file: str, vocos_local_path: str = "/home/milana/app/local_check_data/vocos-mel-24khz"):
        self.config = tomli.load(open(config_path, "rb"))
        self.vocab_file = vocab_file

        # where do we use vocos? 
        vocos = load_vocoder(is_local=True, local_path=vocos_local_path)
        self.tts = self.load_tts_flow()

        self.ref_audio = "/home/milana/app/local_check_data/app/test_en_1_ref_short.wav"
        self.ref_text = self.config["ref_text"]

    def load_tts_flow(self):
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        ckpt_file = self.config.get("ckpt_file", "")
        if not ckpt_file:
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))

        model = load_model(model_cls, model_cfg, ckpt_file, self.vocab_file)
        return model
    
    def generate_audio(self, text: str, source_audio_file: str, save_path: str, remove_silence: bool = True):
        # ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text) # TODO remove and you only the main voice
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text)
        reg2 = r"\[(\w+)\]"

        for text in chunks:
            match = re.match(reg2, text)
            text = re.sub(reg2, "", text)
            gen_text = text.strip()

            audio, final_sample_rate, spectragram = infer_process(self.ref_audio, self.ref_text, gen_text, self.tts)
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            with open(save_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                if remove_silence:
                    remove_silence_for_generated_wav(f.name)
                print(f"Generated audio saved at: {f.name}")
        
    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Voice generation pipeline.', leave=True)):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")            

            if not os.path.exists(file_path):
                # add_pauses(segment.source_file)
                
                self.generate_audio(
                                    segment.translation,
                                    segment.source_file,
                                    file_path,
                                    language
                                    )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation

