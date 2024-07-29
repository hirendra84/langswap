from abc import ABC

from tqdm.auto import tqdm

from src.pipeline_models.models import TranslatedTextedSegment
from TTS.api import TTS
import os
from src.file_repository import FileRepository

import torchaudio
import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from pydub import AudioSegment


def add_pauses(audio_path: str):
    audio, sr = torchaudio.load(audio_path)

    pause_start = torch.zeros((1, sr))
    pause_end = torch.zeros((1, sr))

    audio = torch.cat([pause_start, audio, pause_end], dim=1)
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
        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter_folder}/converter/config.json', device=device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter_folder}/converter/checkpoint.pth')

    def create_speaker(self, audio_path: str):
        se, _ = se_extractor.get_se(audio_path, self.tone_color_converter, vad=True)
        return se

    def merge_enhanced(self, video_translation):
        combined_audio = AudioSegment.empty()

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Merge enhanced pipeline.', leave=True)):
            audio_segment = AudioSegment.from_file(segment.source_file)
            combined_audio += audio_segment

            save_path = video_translation.background_audio["vocals.wav"].file_path
            combined_audio.export(save_path.replace("vocals", "vocals_enhanced"), format="wav")

    def voice_conversion_pipeline(self,
                            video_translation,
                            temp_folder,
                            source_lang):

        self.merge_enhanced(video_translation)
        clean_audio_speaker = video_translation.background_audio["vocals.wav"].file_path.replace("vocals", "vocals_enhanced")
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
                tts_model_id="tts_models/multilingual/multi-dataset/xtts_v2",
                language="en",
                device="cuda"):
        gpu = True if device == "cuda" else False
        self.tts = TTS(tts_model_id, gpu=gpu)
        self._file_repository = file_repository
        self.lang = language
    
    def generate_audio(self, text: str, source_audio_path: str, save_path: str):
        """
        Generates and styles audio, saves according to the path.
        : param style: whether voice conversion should be applied
        """
        self.tts.tts_to_file(text=text, file_path=save_path, speaker_wav=source_audio_path, language=self.lang)
        
    def tts_pipeline(self, video_translation, temp_folder):
        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc='Voice generation pipeline.', leave=True)):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")            

            if not os.path.exists(file_path):
                self.generate_audio(
                                    segment.translation,
                                    segment.source_file,
                                    file_path
                                    )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation