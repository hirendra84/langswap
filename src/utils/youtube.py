import re

from pytube import YouTube


async def _get_suitable_yt_stream(yt: YouTube):
    streams = yt.streams. \
        filter(mime_type="video/mp4", type="video", progressive=True)  # TODO: choose codec
    resolutions_priority = ["1080p", "720p", "480p", "360p", "240p", "144p"]
    suitable_stream = None
    for res in resolutions_priority:
        streams_with_res = streams.filter(res=res)
        if not streams_with_res:
            continue
        for stream in streams_with_res:
            if not (60 >= stream.fps >= 24):
                continue
            suitable_stream = stream
        if suitable_stream:
            break
    if suitable_stream is None:
        raise ValueError('Can\'t find suitable video format')
    return suitable_stream


async def _validate_yt_link(link: str) -> str:
    regexp = re.compile(
        '^(?:https?:\\/\\/)?(?:www\\.)?'
        '(?:youtu\\.be\\/|youtube\\.com\\/'
        '(?:embed\\/|v\\/|watch\\?v=|watch\\?.+&v=))'
        '((\\w|-){11})?$'
    )
    match = regexp.match(link)
    try:
        yt_link = match.group()
    except AttributeError:
        raise

    return yt_link