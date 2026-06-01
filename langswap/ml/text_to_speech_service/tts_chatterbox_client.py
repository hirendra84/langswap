import os

from tqdm.auto import tqdm

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from langswap.file_repository import FileRepository
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.ml.text_to_speech_service.utils import add_pauses, merge_speaker_files


class ChatterboxClient:
    def __init__(
        self,
        file_repository: FileRepository,
        # tts_model_path="./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2",
        device="cuda",
    ):
        self.device = device
        self.sample_rate = 24000

        # self.tts_model_path = os.path.abspath(tts_model_path)

        self._file_repository = file_repository

        self.model = None
        self.load_models()

    def load_models(self):
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model = None

    def generate_audio(
        self, text: str, source_audio_file: str, source_text: str, save_path: str, language: str
    ):
        """
        Generates audio without voice conversion.
        """
        wav = self.model.generate(text, 
                    audio_prompt_path=source_audio_file, language_id=language)
        ta.save(save_path, wav, self.sample_rate)


    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        language = map_language_to_code(language, "whisper")

        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice generation pipeline.",
                leave=True,
            )
        ):  
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")
            if not os.path.exists(file_path):
                source_file = segment.source_file
                if segment.end - segment.start < 4:
                    source_file_updated = segment.source_file.replace(".wav", "_extended.wav")
                    merge_speaker_files(video_translation,
                                    segment.speaker,
                                    idx,
                                    source_file_updated
                                    )
                    source_file = source_file_updated
                else:
                    add_pauses(segment.source_file)
            
                self.generate_audio(
                    text = segment.translation,
                    source_audio_file = source_file,
                    source_text = "", # Chatterbox doesn't use source text
                    save_path = file_path,
                    language = language
                )
                
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation


if __name__ == "__main__":
    # Minimal smoke test for generate_audio using a dummy model to avoid heavy downloads.
    import os
    from datetime import datetime


    client = ChatterboxClient(file_repository=None, device="cuda")
    client.load_models()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    assets_dir = os.path.join(root_dir, "assets")
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    src = os.path.join(assets_dir, "sample_audio.wav")
    text_path = os.path.join(assets_dir, "sample_text_en.txt")
    assert os.path.exists(src), f"Missing reference audio: {src}"
    assert os.path.exists(text_path), f"Missing text file: {text_path}"
    with open(text_path, "r", encoding="utf-8") as f:
        sample_text = f.read().strip()
    assert sample_text, "Sample text is empty"

    out_filename = f"chatterbox_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    out = os.path.join(data_dir, out_filename)
    client.generate_audio(sample_text, src, "", out, "en")

    # Basic file assertions
    assert os.path.exists(out) and os.path.getsize(out) > 0, "Output WAV not created"

    # Audio content assertions
    wav, sr = ta.load(out)
    assert sr == client.sample_rate, f"Unexpected sample rate {sr}, expected {client.sample_rate}"
    assert wav.numel() > 0 and wav.abs().max().item() > 1e-5, "Audio seems silent (max amplitude too low)"
    rms = (wav.pow(2).mean().sqrt().item())
    assert rms > 1e-5, f"Audio RMS too low: {rms}"
    duration = wav.shape[-1] / sr
    assert 2.0 <= duration <= 10.0, f"Unexpected duration: {duration}s"

    print(f"ChatterboxClient generate_audio smoke test passed. Output saved to: {out}")
    print(f"- Sample rate: {sr} Hz, duration: {duration:.3f}s, RMS: {rms:.6f}, peak: {wav.abs().max().item():.6f}")

