# Streaming / Chunked Dubbing — Design Doc

Branch: `ilya/1-container`
Status: proposal (pre-code)
Goal: **minimize time-to-first-segment (TTFS)**, not total runtime. Instead of
returning one final MP4 after ~40s, start emitting watchable dubbed video within
seconds and stream the rest segment-by-segment while the viewer is already
watching.

---

## 1. Why this is possible (where the time goes)

Warm serial floor on `danger.mp4` (16s clip):

| stage        | cost   | nature                                  |
|--------------|--------|-----------------------------------------|
| demucs       | ~7s    | once, whole-audio (vocals + background) |
| ASR          | ~7s    | whole-audio (sherpa-onnx 0.6B, **CPU**) |
| translate    | ~5s    | batch over all segments                 |
| TTS synth    | ~14s   | **per-segment, sequential** (~2.3s/seg) |
| merge        | ~6s    | whole-timeline concat + global stretch  |
| **total**    | ~40s   |                                         |

The two dominant costs — **TTS (14s)** and **merge (6s)** — are inherently
*per-segment* and *sequential over the timeline*. That is exactly the work that
streaming hides behind playback: the viewer watches segment N while we synthesize
N+1, N+2, …. So:

```
TTFS  ≈  (front matter: demucs + ASR)  +  (first chunk: translate + TTS + mux)
      ≈  ~14s              +              ~4s          ≈  ~18s   (v1, safe)
```

vs. ~40s today. With the front-matter optimization in §7 (run ASR on raw audio
in parallel with demucs), TTFS drops toward **~10–11s**. After the first chunk,
production roughly keeps pace with playback (≈2.3s compute per ≈3s of content).

The point is not that any single stage gets faster — it's that **the user starts
watching at TTFS instead of at TOTAL**.

---

## 2. Delivery mechanism — decision

Two candidates were researched (HLS-playlist-on-S3 vs chunked-HTTP fMP4).

**Decision: HLS EVENT playlist + fMP4/CMAF segments written to S3, played with
hls.js.** Rationale:

- **Serverless-native.** Each segment is one independent S3 PUT; the manifest is
  rewritten after each. The worker never holds a socket open for the whole job,
  and viewer lifetime is decoupled from worker lifetime. This fits the existing
  Modal/RunPod model (which is already poll-based: `run`→`call_id`→`status`).
- **Broad playback + seek/rewind.** hls.js everywhere, native HLS on Safari/iOS.
  An EVENT playlist grows append-only; viewers can rewind to the start.
- **Reuses our infra.** We already push everything to S3 via `RemoteFileRepository`.

Chunked-HTTP fMP4 → MSE has marginally lower TTFS but loses seeking/ABR, ties a
worker to the connection for the whole job, and has weak Safari support. We keep
it out of v1. (Modal `remote_gen()` / RunPod `/stream` are noted as a future
"lobby preview" option only.)

Playlist shape (EVENT, fMP4, independent segments):

```
#EXTM3U
#EXT-X-VERSION:7
#EXT-X-TARGETDURATION:6
#EXT-X-PLAYLIST-TYPE:EVENT
#EXT-X-INDEPENDENT-SEGMENTS
#EXT-X-MAP:URI="init.mp4"
#EXTINF:3.840,
seg00000.m4s
#EXTINF:4.200,
seg00001.m4s
...                      <- appended as each chunk lands
#EXT-X-ENDLIST           <- written only when the dub finishes (unlocks full seek)
```

The init segment (`ftyp`+`moov`) is written once; each media segment is a
self-contained `moof`+`mdat` starting on a keyframe → independently decodable.

---

## 3. Chunking strategy — the hard part

### 3.1 Where to cut

- **Never mid-word.** Cut only on existing **inter-segment pauses** — the
  forced-aligner/VAD boundaries already encoded in `TextedSegment.start/end`.
- A *chunk* = one or more consecutive dubbing segments. Chunk size is chosen to
  trade TTFS (smaller = faster first frame) against overhead (more S3 PUTs,
  more keyframe waste). Target ~3–6s of source timeline per chunk; **first chunk
  deliberately small (1 segment)** to minimize TTFS, then grow.

### 3.2 The video-keyframe tension (important)

`-c:v copy` can only cut a clean, independently-decodable segment **at an
existing keyframe**. But dubbing-segment boundaries fall at arbitrary VAD pauses,
which will *not* line up with video keyframes. "Force keyframes at boundaries"
implies re-encoding, which we want to avoid.

Resolution — **let video keyframes be the master grid, snap audio to them:**

1. Probe source keyframe PTS once up front (`ffprobe -select_streams v -skip_frame
   nokey -show_frames`).
2. For each chunk, choose the cut point = the **nearest source keyframe at/after**
   the chunk's intended pause boundary. Cut video with `-c:v copy` there.
3. The chunk's audio is then **padded with silence / trimmed to exactly match the
   keyframe-aligned video-slice duration.** Because the boundary lands inside an
   inter-segment pause (silence), the small slop between the pause boundary and
   the keyframe is inaudible — it's just silence length.

This makes **video-slice duration the per-chunk master clock** and is the core
anti-drift mechanism (§4). Fallback: if the source GOP is too sparse (keyframes
>~6s apart) to give good TTFS, do a **one-time, up-front re-encode** with a fixed
small GOP (`-g`, `-force_key_frames`) — a single fast pass (GPU NVENC if
available), *not* per-chunk re-encoding. This is gated behind a flag
(`LANGSWAP_STREAM_REKEY`) and off by default.

### 3.3 Demucs & continuous background (no chunking of separation)

- **Demucs runs once, up front, on the full audio.** It yields vocals (for ASR /
  voice reference) and the background stems (for remux). We never chunk
  separation — chunked demucs produces clicks at seams.
- The continuous background track is **sliced per chunk** for remux, but the
  slices come from one continuous separation, so seams are clean. We add a small
  **crossfade/overlap at chunk seams** when slicing background to be safe against
  boundary clicks.

---

## 4. A/V sync across chunks — running offset, no global stretch

Today both `merge_timestamps_speedup` and `merge_timestamps_stretch_whole` end
with a **global** `time_stretch` over the *entire* concatenated track
(`rate = total_generated_len / total_source_len`). **That global pass is the one
thing that cannot stream** — it needs every segment before it can run.

### 4.1 Per-segment fitting (drop the global pass)

Move the fitting fully per-segment (the per-segment logic already exists in
`merge_timestamps_speedup` lines 183–241):

1. `change_pauses()` to match intra-utterance pauses to source (unchanged).
2. If the segment audio is longer than its source window → `time_stretch` that
   **segment** to fit (with the existing "borrow from inter-segment pause" rule).
3. If shorter → pad with silence.
4. **No global stretch.** Each segment is self-fitted to its source window, so
   the per-segment error never needs a global correction.

### 4.2 Cumulative timeline / drift

Maintain an explicit **running offset** anchored to the *video* timeline, not to
measured audio lengths (chaining off measured lengths is what accumulates drift):

```
chunk_video_start[k]   = nearest_keyframe(intended_pause_boundary[k])   # from ffprobe, exact
chunk_video_dur[k]     = chunk_video_start[k+1] - chunk_video_start[k]
chunk_audio[k]         = concat(fitted segments in chunk k)
chunk_audio[k]         = pad_or_trim(chunk_audio[k], to = chunk_video_dur[k])  # re-anchor!
```

Each chunk re-anchors audio to the real video PTS, so **drift cannot accumulate**
across hundreds of chunks. `aresample=async=1` is used only as a final polish on
the muxed audio, never as the primary sync mechanism. In fMP4 terms, each
fragment's `tfdt` baseMediaDecodeTime is set from the running offset.

This per-segment merge is the streaming-safe variant of `merge_timestamps_speedup`;
the batch `_merge()` path is **left untouched**.

---

## 5. Pipelining — overlap stages across chunks

The whole point: while TTS synthesizes chunk N, translate/ASR works on N+1.

Async producer/consumer with **bounded, ordered** queues (backpressure; emit in
playlist order even if a later chunk's compute finishes first):

```
[demucs once] ─┐
[ASR once]   ──┴─► segments ─► Q_translate ─► Q_tts ─► Q_merge ─► Q_mux ─► S3 + manifest
                    (chunker)     stage          stage     stage      stage
```

- Each stage is a coroutine pulling from its input queue, pushing to its output
  queue. `asyncio.Queue(maxsize=K)` gives backpressure so a fast stage can't run
  away from a slow one (TTS is the bottleneck → everything upstream naturally
  blocks on it, which is fine).
- GPU stages (TTS, demucs) are serialized via a single GPU worker / lock; CPU ASR
  can overlap GPU work. Per-segment processing is preserved — **no batching**
  (batching blocks streaming and is vetoed).
- **Ordered emission:** the mux stage emits strictly in chunk index order; a
  chunk that finishes early waits its turn before being appended to the manifest.

v1 keeps ASR as a single up-front pass (it gives *all* segment boundaries +
diarization at once, which is simplest and correct). Translate→TTS→merge→mux are
the pipelined stages. (Streaming ASR is a v2 lever, §7.)

---

## 6. Code structure

**New, additive — batch path stays intact:**

- `langswap/streaming.py` — the async orchestrator. Reuses existing stage
  managers (`SpeechToTextManager`, `TranslationManager`, `TextToSpeechManager`,
  `VideoDubbingManager`, `DemucsClient`, `FFmpegClient`) via the warm `model_pool`.
  Exposes an **async generator** `stream_dub(input) -> yields {seg_index, s3_url,
  duration, manifest_url}` and writes segments + manifest to S3 as it goes.
- `VideoDubbingManager.fit_segment(...)` — extracted per-segment fit (the §4.1
  logic, no global stretch). `merge_timestamps_speedup` can be refactored to call
  it, but its external behavior (incl. the global pass) is preserved for batch.
- `FFmpegClient` additions:
  - `probe_keyframes(path) -> list[float]`
  - `init_fmp4_segment(...)` / `write_fmp4_segment(video_slice, audio, idx)` —
    `-c:v copy -hls_segment_type fmp4 -force_key_frames`-aware single-segment mux.
  - `HlsManifest` helper (write/append EVENT playlist, finalize with `ENDLIST`).
- `langswap/streaming_hls/` (or reuse S3) — manifest + segment naming under the
  job's S3 prefix: `{public_id}/hls/{init.mp4, seg%05d.m4s, index.m3u8}`.

**Entrypoints (additive):**

- `serverless.py` — add a generator handler variant (RunPod generator + `/stream`)
  that yields per-segment events; keep the existing `handler` untouched.
- `modal_app.py` — add `Dubber.stream_dub` (`@modal.method` returning a generator)
  + a `stream` FastAPI endpoint using `StreamingResponse(... .remote_gen())`. The
  segments themselves go to S3; the stream just carries manifest-ready events.
- `langswap/api.py` — `process_translation` (batch) is **not modified**; add
  `stream_translation(input)` async generator alongside it.

**Local eval:**

- `tools/hls_preview.html` — a one-file hls.js page that loads the EVENT playlist
  (local or S3 URL) so we can eyeball seam quality and A/V sync as segments land.
- `tools/run_streaming_local.py` — runs `stream_dub` against a local file, writes
  HLS to a local dir, prints TTFS + per-chunk timings.

---

## 7. Phasing (incremental PR plan)

**Phase 0 — plumbing (this PR's spine).** Demucs+ASR up front (unchanged),
then a *serial* chunk loop: per chunk → fit segment → mux fMP4 → write to S3 →
append manifest. No async overlap yet. Proves the muxing/manifest/playback chain
end-to-end with correct A/V sync. Local hls.js page works. This alone gives
TTFS ≈ demucs+ASR+translate+TTS(first) because chunks emit as produced.

**Phase 1 — overlap.** Wrap the chunk stages in the bounded async pipeline (§5)
so TTS(N) overlaps translate(N+1). Ordered emission. This is where the per-chunk
production-keeps-pace-with-playback property kicks in.

**Phase 2 — front-matter TTFS.** Optional levers, measured & flagged:
- Run **ASR on raw audio in parallel with demucs** (CPU ASR ∥ GPU demucs) →
  collapses the ~14s front matter toward ~7s. Validate ASR quality vs.
  vocals-input before defaulting on.
- Streaming/windowed ASR to emit first segments before the whole file is
  transcribed (push TTFS to single digits).
- `LANGSWAP_STREAM_REKEY` one-time re-encode for deterministic small GOP when
  source keyframes are too sparse.

**Phase 3 — serverless stream endpoints** (Modal `stream`, RunPod generator).

---

## 8. Risks & open questions

- **Keyframe sparsity.** If source GOPs are long, chunk granularity is coarse and
  TTFS suffers without the §3.2 re-encode fallback. Need to measure real inputs.
- **Background seam clicks.** Slicing one continuous demucs background per chunk
  should be clean, but verify crossfade at seams.
- **AAC priming delay** per independently-encoded segment can inject small audio
  offsets → encode audio against the master timeline / account for edit lists.
- **Per-segment vs global stretch quality.** Dropping the global pass may make a
  few segments slightly faster/slower than the old global-averaged result.
  Eyeball on the hls.js page; the per-segment fit is generally *more* locally
  accurate.
- **EVENT-playlist seek before `ENDLIST`.** Some players restrict seeking past
  the live edge until the dub finishes — acceptable UX, documented.
- **Warm pool + concurrency.** The async pipeline must not load models twice;
  all stages go through `get_or_create`. GPU stages serialized.

---

## 9. Checklist

### Phase 0 — streaming spine (serial)
- [ ] `FFmpegClient.probe_keyframes(path)` via ffprobe; unit-eyeball on danger.mp4.
- [ ] `FFmpegClient.write_fmp4_segment()` — single keyframe-aligned fMP4 segment,
      `-c:v copy`, audio replaced; produces shared `init.mp4` once.
- [ ] `HlsManifest` writer — EVENT playlist, `EXT-X-MAP`, append, `ENDLIST`.
- [ ] `VideoDubbingManager.fit_segment()` — per-segment fit, **no global stretch**;
      running-offset re-anchor to keyframe-aligned video-slice duration.
- [ ] Demucs once up front; slice continuous background per chunk (+crossfade).
- [ ] `langswap/streaming.py` `stream_dub()` serial loop → S3 + manifest, yields events.
- [ ] `tools/hls_preview.html` (hls.js) + `tools/run_streaming_local.py`.
- [ ] Verify A/V sync & seams on danger.mp4; record TTFS.
- [ ] Batch path (`process_translation`, `_merge`) byte-for-byte unchanged.

### Phase 1 — overlap
- [ ] Bounded `asyncio.Queue` stages: translate → tts → merge → mux.
- [ ] Single GPU worker/lock; CPU ASR overlaps; per-segment preserved (no batching).
- [ ] Ordered emission (chunk i appended only after i-1).
- [ ] Backpressure verified (TTS bottleneck blocks upstream cleanly).

### Phase 2 — front-matter TTFS (flagged, measured)
- [ ] ASR-on-raw ∥ demucs option; quality A/B; flag.
- [ ] (stretch) windowed/streaming ASR.
- [ ] `LANGSWAP_STREAM_REKEY` one-time small-GOP re-encode fallback.

### Phase 3 — serverless
- [ ] `modal_app.py` `stream` endpoint (`remote_gen` + `StreamingResponse`).
- [ ] `serverless.py` generator handler (`return_aggregate_stream`) + `/stream`.
- [ ] Manifest/segment S3 layout + lifecycle/cleanup.

---

## 10. Key references
- ffmpeg fMP4 HLS / `-force_key_frames` / `-c:v copy` remux; `aresample=async=1`.
- HLS EVENT playlist (append-only, `EXT-X-MAP`, `EXT-X-INDEPENDENT-SEGMENTS`,
  `ENDLIST`); CMAF `moof`+`mdat` independent decodability; `tfdt` anchor.
- hls.js EVENT/live config (`liveSyncDurationCount`, `lowLatencyMode`).
- Modal `remote_gen()` + `StreamingResponse`; RunPod generator handler + `/stream`.
</content>
</invoke>
