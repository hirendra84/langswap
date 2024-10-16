import sys
from abc import ABC

from src.pipeline_models.models import TranslatedTextedSegment
from tqdm.auto import tqdm

sys.path.append("/app/coqui")

import os
import sys

import torch
import torchaudio
from coqui.TTS.api import TTS
from src.file_repository import FileRepository
from src.utils.ml_processing.lang2code_mapper import map_language_to_code

sys.path.append("/app/voice_conv")

from pydub import AudioSegment
from voice_conv.openvoice import se_extractor
from voice_conv.openvoice.api import ToneColorConverter


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
        speaker = self.create_speaker(clean_audio_speaker)    

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Voice conversion pipeline.', leave=True)):            
            folder_path, audio_name = os.path.split(segment.source_file)
            audio_save_path = os.path.join(temp_folder, audio_name)
            
            self.tone_color_converter.convert(
                audio_src_path=segment.generated_file,
                src_se=speaker,
                tgt_se=speaker,
                output_path=audio_save_path
            )
        
            video_translation.translated_texts[idx].generated_file = audio_save_path
        return video_translation


class XTTSClient:
    def __init__(self,
                file_repository: FileRepository,
                tts_model_path="./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2",
                device="cuda"):
        gpu = True if device == "cuda" else False
        tts_model_path = os.path.abspath(tts_model_path)

        config_path = "./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2/config.json"
        config_path = os.path.abspath(config_path)
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