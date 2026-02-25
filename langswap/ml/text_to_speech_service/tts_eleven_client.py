import os
import types

from elevenlabs import save
from elevenlabs.client import ElevenLabs
from tqdm import tqdm

from langswap.pipeline_models.models import TranslatedTextedSegment


class ElevenTTSClient:

    def __init__(self, eleven_api_token):
        self.client = ElevenLabs(api_key=eleven_api_token)
        self.sample_rate = 24000
        # If ELEVEN_VOICE_ID is set, skip cloning and use this voice directly.
        self._preset_voice_id = os.environ.get("ELEVEN_VOICE_ID")

    def _make_voice_ref(self, voice_id: str):
        """Thin wrapper so callers can always do voice.voice_id."""
        v = types.SimpleNamespace()
        v.voice_id = voice_id
        return v

    def clone_voice(self, video_translation, voice_descr: str = "", voice_name=""):
        # If a preset voice is configured, use it without cloning.
        if self._preset_voice_id:
            return self._make_voice_ref(self._preset_voice_id)

        audio_files_source = [s.source_file for s in video_translation.translated_texts]
        file_handles = [open(f, "rb") for f in audio_files_source[:24]]
        try:
            voice = self.client.voices.ivc.create(
                name=voice_name or "cloned_voice",
                description=voice_descr,
                files=file_handles,
            )
        finally:
            for fh in file_handles:
                fh.close()

        return voice

    def generate_audio(self, text: str, voice, save_path: str, source_text=None):
        audio = self.client.text_to_speech.convert(
            voice_id=voice.voice_id,
            text=text,
            output_format="mp3_44100_128",
        )
        save(audio, save_path)

    def tts_pipeline(
        self, video_translation, temp_folder: str, language="en"
    ) -> list[TranslatedTextedSegment]:
        cloned = self._preset_voice_id is None
        voice = self.clone_voice(video_translation)

        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice generation pipeline.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.mp3")

            if not os.path.exists(file_path):
                self.generate_audio(
                    segment.translation, voice=voice, save_path=file_path
                )
            video_translation.translated_texts[idx].generated_file = file_path

        if cloned:
            self.client.voices.delete(voice_id=voice.voice_id)
        return video_translation
