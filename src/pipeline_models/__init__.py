import attr


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
    source_url: str
    extracted_audio_url: str = attr.ib(default='')
    vad_filtered_audio_url: str = attr.ib(default='')
    recognized_texts: list[TextedSegment] = attr.field(factory=list)
    translated_texts: list[TranslatedTextedSegment] = attr.field(factory=list)
    processed_video: str = attr.ib(default='')
