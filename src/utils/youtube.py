import io
import re
import urllib

import yt_dlp


def get_yt_stream_and_name(link) -> tuple[io.BytesIO, str]:
    link = _validate_yt_link(link)
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(link, download=False)
        video_url = video_info['url']
        video_title = video_info['title']

    try:
        with urllib.request.urlopen(video_url) as response:
            video_data = io.BytesIO(response.read())

    except Exception as e:
        print(f"Error uploading video to S3: {e}")
        raise

    return video_data, video_title


def _validate_yt_link(link: str) -> str:
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
