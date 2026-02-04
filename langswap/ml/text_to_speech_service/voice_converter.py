import sys

sys.path.append("/app/voice_conv")

from voice_conv.openvoice import se_extractor
from voice_conv.openvoice.api import ToneColorConverter

from pydub import AudioSegment
import os

from tqdm import tqdm


class VoiceToneConverter:
    def __init__(self, ckpt_converter_folder: str, device="cpu"):
        self.ckpt_converter_folder = ckpt_converter_folder
        self.config_path = os.path.join(self.ckpt_converter_folder, "converter/config.json")
        self.checkpoint_path = os.path.join(
            self.ckpt_converter_folder, "converter/checkpoint.pth"
        )
        self.tone_color_converter = None

        self.device = device

        self.speaker = None
    
    def load_models(self,):
        self.tone_color_converter = ToneColorConverter(self.config_path, device=self.device)
        self.tone_color_converter.load_ckpt(self.checkpoint_path)
    
    def __enter__(self):
        self.load_models()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.tone_color_converter = None
        self.speaker = None
    
    def generate_speaker_embedding(self, audio_path: str):
        se, _ = se_extractor.get_se(audio_path, self.tone_color_converter, vad=True)
        return se


    def create_speaker(self, video_translation):
        self.merge_enhanced(video_translation)
    
        cleaned_audio_path = video_translation.background_audio["vocals.wav"].replace(
            "vocals", "vocals_enhanced"
        )
        self.speaker = self.generate_speaker_embedding(cleaned_audio_path)

    def merge_enhanced(self, video_translation):
        combined_audio = AudioSegment.empty()

        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Merge enhanced pipeline.",
                leave=True,
            )
        ):
            audio_segment = AudioSegment.from_file(segment.source_file)
            combined_audio += audio_segment

            save_path = video_translation.background_audio["vocals.wav"]
            combined_audio.export(
                save_path.replace("vocals", "vocals_enhanced"), format="wav"
            )
                

    def voice_conversion_pipeline(self, video_translation, temp_folder, source_lang, use_cashe: bool = True):
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice conversion pipeline.",
                leave=True,
            )
        ):
            folder_path, audio_name = os.path.split(segment.source_file)
            audio_save_path = os.path.join(temp_folder, audio_name)
            
            if not use_cashe or not os.path.exists(audio_save_path):
                
                if self.speaker is None:
                    self.create_speaker(video_translation)
                    
                speaker = self.generate_speaker_embedding(segment.generated_file)

                self.tone_color_converter.convert(
                    audio_src_path=segment.generated_file,
                    src_se=speaker,
                    tgt_se=self.speaker,
                    output_path=audio_save_path,
                )

            video_translation.translated_texts[idx].generated_file = audio_save_path
        return video_translation
