"""Post-processing: trim leading garbage / trailing noise from generated WAVs.

Ported from the `jaitts_tools` project's `trim_leading.py`. Uses faster-whisper
with `word_timestamps=True` to locate the words, aligns the transcript against
the expected spoken text with difflib, then cuts audio before the first matched
word and after the last matched word. Falls back to an energy-based trailing
noise cut when the text match is inconclusive.

A low overall transcript/expected ratio (< 0.35) means the clip is NOT the
expected text - callers should skip trimming entirely (nothing is cut).
"""

import difflib
import threading
from pathlib import Path

import numpy as np

HEADROOM = 0.10  # seconds kept before the first matched word
TAIL_MARGIN = 0.15  # seconds kept after the last matched word
MIN_CUT = 0.30  # don't cut unless there is at least this much to remove
MIN_MATCH_RATIO = 0.35  # below this the transcript != expected text -> skip


def read_wav(path):
    import wave

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = (
            np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        )
    return data, sr


def write_wav(path, data, sr):
    import wave

    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def norm(s):
    return "".join(ch for ch in s if not ch.isspace() and ch not in "!?.,;:()\"'")


def _offset_to_word(words_norm, offset):
    off = 0
    for i, wn in enumerate(words_norm):
        if offset < off + len(wn):
            return i
        off += len(wn)
    return None


def find_cuts(words, expected, min_start_block=4, min_end_block=3):
    """Align transcript words against expected text.

    Returns (start_word_idx, end_word_idx, ratio) - start/end may be None.
    A low overall ratio (< 0.35) means the transcript is NOT the expected text:
    callers should skip trimming entirely.
    """
    exp = norm(expected)
    words_norm = [norm(w.word) for w in words]
    total = "".join(words_norm)
    sm = difflib.SequenceMatcher(None, exp, total, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    ratio = sm.ratio()

    start_word = None
    a0 = [b for b in blocks if b.a == 0 and b.size >= min_start_block]
    if a0:
        best = max(a0, key=lambda b: b.size)
        start_word = _offset_to_word(words_norm, best.b)
    else:
        # No block anchored at the exact start of the expected text. Two very
        # different situations look identical here:
        #   (1) REAL babble prefix ("คอยก้าห้าหายเยี่ย ... เมื่อคืน...") ->
        #       the audio before the match has NO relation to the text, cut it.
        #   (2) first word merely mis-heard ("มัววายผมไป" vs "เมื่อวานผมไป") ->
        #       the words ARE the text, cutting loses real speech (bad).
        # Be conservative so case (2) is NEVER cut: require a long block
        # (>= 8 chars, a real phrase) anchored within the first 10% of the
        # expected text. Babble blocks easily satisfy this; mis-heard-word
        # offsets land further in and get skipped.
        near = [
            b for b in blocks if b.a <= max(4, int(len(exp) * 0.10)) and b.size >= 8
        ]
        if near:
            best = min(near, key=lambda b: (b.b, -b.size))
            start_word = _offset_to_word(words_norm, best.b)

    end_word = None
    full_end = [
        b for b in blocks if b.a + b.size == len(exp) and b.size >= min_end_block
    ]
    if full_end:
        last = max(full_end, key=lambda b: b.b + b.size)
        end_word = _offset_to_word(words_norm, last.b + last.size - 1)
    return start_word, end_word, ratio


def trailing_energy_start(
    data, sr, threshold=0.01, win_s=0.05, min_run_s=1.2, max_gap_s=0.1
):
    """Start time of the last continuous run of loud-ish windows (noise /
    babble) that extends to the end of the file, or None. The run must last
    at least min_run_s seconds (brief dips of max_gap_s are allowed)."""
    win = int(sr * win_s)
    n = len(data) // win
    if n < 2:
        return None
    rms = np.array(
        [np.sqrt(np.mean(data[i * win : (i + 1) * win] ** 2)) for i in range(n)]
    )
    max_gap = max(1, int(max_gap_s / win_s))
    if rms[-1] <= threshold:
        return None
    below = 0
    start = n - 1
    i = n - 1
    while i >= 0:
        if rms[i] <= threshold:
            below += 1
            if below > max_gap:
                break
        else:
            below = 0
            start = i
        i -= 1
    if (n - start) * win_s < min_run_s:
        return None
    return start * win_s


def trim_audio(
    data,
    sr,
    words,
    expected,
    headroom=HEADROOM,
    tail_margin=TAIL_MARGIN,
    min_cut=MIN_CUT,
):
    """Cut leading garbage before the first matched word and trailing audio
    after the last matched word. Returns (trimmed_data, logs).
    """
    if not words:
        return data, []
    start_word, end_word, ratio = find_cuts(words, expected)
    if ratio < MIN_MATCH_RATIO:
        return data, [
            f"SKIP: transcript differs from expected text (match {ratio:.2f}) - nothing cut"
        ]
    cut_start = 0
    cut_end = len(data)
    logs = []

    if start_word is not None:
        t = max(0.0, words[start_word].start - headroom)
        if t >= min_cut:
            cut_start = int(t * sr)
            logs.append(
                f"start: cut {t:.2f}s (first matched word '{words[start_word].word}' @ {words[start_word].start:.2f}s)"
            )

    if end_word is not None:
        t = min(len(data) / sr, words[end_word].end + tail_margin)
        if len(data) / sr - t >= min_cut:
            cut_end = int(t * sr)
            logs.append(
                f"tail: cut {len(data) / sr - t:.2f}s (last matched word '{words[end_word].word}' @ {words[end_word].end:.2f}s)"
            )
    else:
        t_noise = trailing_energy_start(data, sr)
        if t_noise is not None:
            last_end = words[-1].end if words else 0.0
            if last_end <= t_noise - 0.3:
                tail_len = len(data) / sr - t_noise
                if tail_len >= min_cut:
                    cut_end = int(t_noise * sr)
                    logs.append(
                        f"tail: cut {tail_len:.2f}s (noise from {t_noise:.2f}s)"
                    )

    if cut_start > 0 or cut_end < len(data):
        return data[cut_start:cut_end], logs
    return data, []


_whisper_model = None
_whisper_lock = threading.Lock()
_whisper_key = None


def _get_whisper_model(model_size, device, compute_type, download_root):
    """Load (and cache) the faster-whisper model used for trimming."""
    global _whisper_model, _whisper_key
    key = (model_size, device, compute_type, download_root)
    with _whisper_lock:
        if _whisper_model is not None and key == _whisper_key:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed - required for jaitts_tts "
                "audio trimming. Install it or set 'trim_audio: false' in "
                "conf.yaml (jaitts_tts)."
            ) from e

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        _whisper_model = model
        _whisper_key = key
        return model


def trim_wav_file(
    wav_path: str,
    expected_text: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    download_root: Path | None = None,
    headroom: float = HEADROOM,
    tail_margin: float = TAIL_MARGIN,
    min_cut: float = MIN_CUT,
) -> list[str]:
    """Trim a generated WAV in place using faster-whisper word timestamps
    aligned against `expected_text`. Returns a list of log messages
    (empty = nothing was cut)."""
    wav = Path(wav_path)
    if not wav.is_file():
        return [f"SKIP: {wav_path} not found"]

    model = _get_whisper_model(model_size, device, compute_type, download_root)
    data, sr = read_wav(wav)
    segments, _ = model.transcribe(
        str(wav), language="th", beam_size=5, word_timestamps=True
    )
    segments = list(segments)
    words = [w for seg in segments for w in seg.words]
    if not words:
        return ["SKIP: no words transcribed - nothing cut"]

    trimmed, logs = trim_audio(
        data,
        sr,
        words,
        expected_text,
        headroom=headroom,
        tail_margin=tail_margin,
        min_cut=min_cut,
    )
    if not logs:
        return []
    if len(trimmed) == len(data):
        return logs

    write_wav(wav, trimmed, sr)
    return logs
