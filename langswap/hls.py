"""HLS / fMP4 helpers for streaming dubbing (Phase 0).

This module is pure-Python and has no ML / GPU / S3 dependencies, so it can be
unit-tested on its own against a synthetic clip.  Two pieces:

* ``split_fragmented_mp4`` — split a fragmented MP4 produced by ffmpeg
  (``ftyp + moov + (moof + mdat)+``) into the CMAF **init segment**
  (``ftyp + moov``, written once and referenced by ``#EXT-X-MAP``) and the
  **media segment** (``moof + mdat ...``, an independently-decodable ``.m4s``).

* ``HlsManifest`` — an append-only HLS **EVENT** playlist writer.  Segments are
  added as they are produced; ``finalize()`` writes ``#EXT-X-ENDLIST`` which
  flips the player from live to a fully-seekable VOD asset.

See docs/streaming_dubbing_design.md §2.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# fMP4 box parsing
# ---------------------------------------------------------------------------

def _iter_boxes(data: bytes):
    """Yield ``(box_type, start, end)`` for each top-level ISO-BMFF box.

    Handles the 32-bit size, the ``size == 1`` 64-bit largesize escape, and the
    ``size == 0`` "to end of file" case.  Only top-level boxes are walked, which
    is all we need to find the ftyp/moov/moof boundaries.
    """
    offset = 0
    n = len(data)
    while offset + 8 <= n:
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        box_type = data[offset + 4:offset + 8].decode("latin-1")
        header = 8
        if size == 1:
            # 64-bit largesize in the 8 bytes following the type
            size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
            header = 16
        elif size == 0:
            size = n - offset  # extends to EOF
        if size < header or offset + size > n:
            break  # truncated / malformed; stop walking
        yield box_type, offset, offset + size
        offset += size


def split_fragmented_mp4(frag_path: str) -> Tuple[bytes, bytes]:
    """Split a fragmented MP4 into ``(init_bytes, media_bytes)``.

    ``init`` = ``ftyp`` + ``moov`` (CMAF init segment).
    ``media`` = everything after ``moov`` (the ``moof``/``mdat`` fragments),
    with any trailing ``mfra`` random-access index dropped — it's optional and
    only bloats every media segment.
    """
    with open(frag_path, "rb") as f:
        data = f.read()

    init_end = None
    media_start = None
    media_end = len(data)
    for box_type, start, end in _iter_boxes(data):
        if box_type == "moov":
            init_end = end
            media_start = end
        elif box_type == "mfra":
            media_end = min(media_end, start)

    if init_end is None or media_start is None:
        raise ValueError(
            f"{frag_path} is not a fragmented MP4 (no moov box found). "
            "Ensure ffmpeg ran with -movflags +empty_moov+frag_keyframe."
        )

    init_bytes = data[:init_end]
    media_bytes = data[media_start:media_end]
    return init_bytes, media_bytes


# ---------------------------------------------------------------------------
# HLS EVENT playlist
# ---------------------------------------------------------------------------

@dataclass
class HlsManifest:
    """Append-only HLS EVENT playlist writer (fMP4 segments).

    The playlist is rewritten in full on every append — HLS requires
    ``#EXT-X-TARGETDURATION`` to be >= the longest segment, so we recompute the
    header each time.  This is cheap (a few hundred bytes) and keeps the file
    valid for a player polling it mid-production.
    """

    playlist_path: str
    init_uri: str = "init.mp4"
    version: int = 7
    segments: List[Tuple[str, float]] = field(default_factory=list)  # (uri, duration_s)
    _finalized: bool = False

    def add_segment(self, uri: str, duration: float) -> None:
        self.segments.append((uri, float(duration)))
        self._write()

    def finalize(self) -> None:
        """Append #EXT-X-ENDLIST — the dub is complete, asset is now seekable."""
        self._finalized = True
        self._write()

    def _render(self) -> str:
        target = max((d for _, d in self.segments), default=1.0)
        lines = [
            "#EXTM3U",
            f"#EXT-X-VERSION:{self.version}",
            f"#EXT-X-TARGETDURATION:{int(math.ceil(target))}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
            "#EXT-X-INDEPENDENT-SEGMENTS",
            f'#EXT-X-MAP:URI="{self.init_uri}"',
        ]
        for uri, duration in self.segments:
            lines.append(f"#EXTINF:{duration:.3f},")
            lines.append(uri)
        if self._finalized:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def _write(self) -> None:
        tmp = self.playlist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self._render())
        os.replace(tmp, self.playlist_path)  # atomic for concurrent pollers
