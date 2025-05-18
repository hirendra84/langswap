import os
from abc import ABC

from elevenlabs import save, Voice
from elevenlabs.client import ElevenLabs
from tqdm import tqdm

from src.pipeline_models.models import TranslatedTextedSegment


class ElevenTTSClient:

    def __init__(self, eleven_api_token):
        self.client = ElevenLabs(api_key=eleven_api_token)  # MOVE HIDDEN ELEVEN_API_KEY
        self.sample_rate = 24000

    def clone_voice(self, video_translation, voice_descr: str = "", voice_name=""):
        audio_files_source = []

        for f_sample in video_translation.translated_texts:
            audio_files_source.append(f_sample.source_file)

        voice = self.client.clone(
            name=voice_name, description=voice_descr, files=audio_files_source[:24]
        )
        
        return voice

    def generate_audio(self, text: str, source_text, voice, save_path: str):
        audio = self.client.generate(text=text, voice=voice)
        save(audio, save_path)

    def tts_pipeline(
        self, video_translation, temp_folder: str, language="en"
    ) -> list[TranslatedTextedSegment]:
        # create the voice
        voice = self.clone_voice(video_translation)

        # generate audio one step in a time
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
                    segment.translation, voice=voice, save_path=file_path
                )
            video_translation.translated_texts[idx].generated_file = file_path
        self.client.voices.delete(voice.voice_id)
        return video_translation
