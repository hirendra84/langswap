import attr
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union, Dict, Literal

@attr.s(auto_attribs=True)
class RemoteFile:
    name: str | None = attr.ib(default=None)
    file_path: str | None = attr.ib(default=None)
    s3_url: str | None = attr.ib(default=None)


@attr.s(auto_attribs=True)
class TextedSegment:
    text: str
    start: float
    end: float
    speaker: str


@attr.s(auto_attribs=True)
class TranslatedTextedSegment:
    text: str
    start: float
    end: float
    translation: str
    source_file: str
    generated_file: str
    speaker: str


@attr.s(auto_attribs=True)
class VideoTranslation:
    public_id: str
    source_file: RemoteFile | None = attr.ib(default=None)
    extracted_audio: RemoteFile | None = attr.ib(default=None)
    vad_filtered_audio: RemoteFile | None = attr.ib(default=None)
    background_audio: dict[str, RemoteFile] = attr.field(factory=dict)
    recognized_texts: list[TextedSegment] = attr.field(factory=list)
    translated_texts: list[TranslatedTextedSegment] = attr.field(factory=list)
    processed_video: RemoteFile | None = attr.ib(default=None)


@dataclass
class TranslationPipelineConfig:
    source_lang: str
    target_lang: str
    source_video_path: Union[Path, str]
    base_dir: Union[Path, str]
    public_id: str
    voice_conv: bool = field(default=False)
    num_speakers: int = field(default=1)
    device: str = field(default="cuda")
    name: str = field(default="example")
    dubbing_algo: Literal["speedup", "pause_based", "stretch_whole"] = field(default="speedup")
    tts_model: Literal["xtts", "f5tts", "elevenlabs"] = field(default="xtts")
    eleven_api_token: str = field(default=None)
    

@dataclass
class TraslationUpdate:
    index: int
    text: str