import attr


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


@attr.s(auto_attribs=True)
class TranslatedTextedSegment:
    text: str
    start: float
    end: float
    translation: str


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
