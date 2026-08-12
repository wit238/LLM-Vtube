"""Qwen3-TTS (DashScope) TTS engine.

Cloud TTS via Alibaba Cloud Model Studio (DashScope) qwen-tts models
(e.g. qwen3-tts-flash-2025-11-27). No GPU needed - works locally and on
Railway. Long replies are split into per-call chunks (the API has a text
length limit) and the resulting WAVs are concatenated into one file.
"""

import os
import re
import time
from pathlib import Path

import httpx
from loguru import logger

from .tts_interface import TTSInterface

_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
_DEFAULT_MODEL = "qwen3-tts-flash-2025-11-27"
_DEFAULT_VOICE = "Cherry"
# Keep the default below the API's per-call text limit.
_DEFAULT_MAX_CHARS = 1200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:。！？；：…\n])\s+")


class TTSEngine(TTSInterface):
    """Qwen3-TTS via the DashScope SpeechSynthesizer API."""

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        voice: str = _DEFAULT_VOICE,
        base_url: str = _DEFAULT_BASE_URL,
        max_chars: int = _DEFAULT_MAX_CHARS,
        timeout: float = 120.0,
        trim_audio: bool = True,
        trim_model: str = "small",
        trim_device: str = "cpu",
        trim_compute_type: str = "int8",
        trim_download_root: str = "models/whisper",
        trim_headroom: float = 0.10,
        trim_tail_margin: float = 0.15,
        trim_min_cut: float = 0.30,
        fallback_tts: str = "",
        fallback_voice: str = "th-TH-PremwadeeNeural",
        fallback_pitch: str = "+10Hz",
        fallback_rate: str = "-14%",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.max_chars = int(max_chars)
        self.timeout = float(timeout)
        self.trim_audio = bool(trim_audio)
        self.trim_model = trim_model
        self.trim_device = trim_device
        self.trim_compute_type = trim_compute_type
        self.trim_download_root = trim_download_root
        self.trim_headroom = float(trim_headroom)
        self.trim_tail_margin = float(trim_tail_margin)
        self.trim_min_cut = float(trim_min_cut)
        self.fallback_tts = fallback_tts
        self.fallback_voice = fallback_voice
        self.fallback_pitch = fallback_pitch
        self.fallback_rate = fallback_rate
        self._fallback_engine = None

    def _get_fallback_engine(self):
        """Lazily build the fallback TTS engine (e.g. edge_tts)."""
        if self._fallback_engine is None and self.fallback_tts:
            if self.fallback_tts == "edge_tts":
                from .edge_tts import TTSEngine as EdgeTTSEngine

                self._fallback_engine = EdgeTTSEngine(
                    voice=self.fallback_voice or "th-TH-PremwadeeNeural",
                    pitch=self.fallback_pitch,
                    rate=self.fallback_rate,
                )
                logger.warning(f"qwen_tts: fallback TTS enabled -> {self.fallback_tts}")
            else:
                logger.warning(
                    f"qwen_tts: unknown fallback_tts '{self.fallback_tts}' "
                    f"(supported: 'edge_tts')"
                )
        return self._fallback_engine

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into pieces of at most max_chars on sentence boundaries."""
        if len(text) <= self.max_chars:
            return [text]
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        chunks: list[str] = []
        cur = ""
        for sent in sentences:
            if len(sent) > self.max_chars:
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.extend(
                    sent[i : i + self.max_chars]
                    for i in range(0, len(sent), self.max_chars)
                )
                continue
            cand = (cur + " " + sent).strip() if cur else sent
            if len(cand) > self.max_chars:
                chunks.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        return chunks

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        if not text or not text.strip():
            raise ValueError("Empty text for Qwen3-TTS synthesis")
        if not self.api_key:
            raise RuntimeError(
                "Qwen3-TTS: no API key - set 'api_key' in conf.yaml "
                "(qwen_tts) or the DASHSCOPE_API_KEY environment variable"
            )

        out_file = self.generate_cache_file_name(file_name_no_ext, "wav")
        chunks = self._chunk_text(text)

        logger.info(
            f"qwen_tts: '{self.model}' voice='{self.voice}' "
            f"({len(text)} chars -> {len(chunks)} call(s))"
        )
        t0 = time.time()
        try:
            part_paths = self._synth_chunks(chunks)
            if len(part_paths) == 1:
                Path(part_paths[0]).replace(out_file)
            else:
                self._concat_wavs(part_paths, out_file)
                for p in part_paths:
                    Path(p).unlink(missing_ok=True)
            logger.info(
                f"qwen_tts: got WAV in {time.time() - t0:.1f}s "
                f"({len(chunks)} chunk(s)) -> {out_file}"
            )

            if self.trim_audio:
                self._trim_wav(out_file, text)

            return out_file
        except (ConnectionError, httpx.HTTPError, RuntimeError) as e:
            # Leave the temp part files for debugging; they are tiny.
            logger.warning(f"qwen_tts: {e}")
            engine = self._get_fallback_engine()
            if engine is not None:
                logger.warning(
                    f"qwen_tts: falling back to {self.fallback_tts} "
                    f"(total so far {time.time() - t0:.1f}s)"
                )
                try:
                    result = engine.generate_audio(text, file_name_no_ext)
                except Exception as fe:
                    logger.error(f"qwen_tts: fallback TTS also failed: {fe}")
                else:
                    if result:
                        logger.info(
                            f"qwen_tts: fallback {self.fallback_tts} produced "
                            f"{result} in {time.time() - t0:.1f}s"
                        )
                        return result
                    logger.error(
                        "qwen_tts: fallback TTS returned no audio - re-raising"
                    )
            raise

    def _synth_chunks(self, chunks: list[str]) -> list[str]:
        """One DashScope call per chunk; returns the saved part WAV paths."""
        import dashscope

        dashscope.base_http_api_url = self.base_url
        from dashscope.audio.qwen_tts import SpeechSynthesizer

        part_paths = []
        for i, chunk in enumerate(chunks):
            logger.info(
                f"qwen_tts: calling API for chunk {i + 1}/{len(chunks)} "
                f"({len(chunk)} chars)"
            )
            t1 = time.time()
            try:
                response = SpeechSynthesizer.call(
                    model=self.model,
                    api_key=self.api_key,
                    text=chunk,
                    voice=self.voice,
                )
            except Exception as e:
                raise ConnectionError(f"Qwen3-TTS API request failed: {e}") from e

            if response.status_code != 200:
                raise RuntimeError(
                    f"Qwen3-TTS API error ({response.status_code}): "
                    f"{getattr(response, 'message', '')}"
                )
            url = response.output["audio"]["url"]
            part_dir = Path("cache")
            part_dir.mkdir(exist_ok=True, parents=True)
            part = part_dir / f"qwen_part_{i:02d}.wav"
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as c:
                    resp = c.get(url)
                    resp.raise_for_status()
                Path(part).write_bytes(resp.content)
            except Exception as e:
                raise ConnectionError(f"Failed to download Qwen3-TTS audio: {e}") from e
            logger.info(
                f"qwen_tts: chunk {i + 1} -> {len(resp.content)} bytes "
                f"({time.time() - t1:.1f}s)"
            )
            part_paths.append(part)
        return part_paths

    def _concat_wavs(self, part_paths: list[str], out_file: str) -> None:
        """Concatenate WAV parts with pydub (pure-python for WAV, no ffmpeg)."""
        from pydub import AudioSegment

        combined = AudioSegment.silent(duration=0)
        for p in part_paths:
            combined = combined + AudioSegment.from_wav(p)
        combined.export(out_file, format="wav")

    def _trim_wav(self, out_file: str, text: str) -> None:
        """Post-process the generated WAV: cut leading garbage / trailing
        noise with faster-whisper word timestamps (same method as the
        jaitts_tools project's trim_leading.py)."""
        try:
            from ..utils.audio_trim import trim_wav_file, read_wav
        except Exception as e:
            logger.warning(f"qwen_tts: audio_trim unavailable, skipping trim: {e}")
            return
        try:
            from pathlib import Path as _Path

            out = _Path(out_file)
            data, sr = read_wav(out)
            if len(data) / sr < 0.5:
                logger.info("qwen_tts: clip too short for trim, skipping")
                return
            logs = trim_wav_file(
                out_file,
                text,
                model_size=self.trim_model,
                device=self.trim_device,
                compute_type=self.trim_compute_type,
                download_root=_Path(self.trim_download_root)
                if self.trim_download_root
                else None,
                headroom=self.trim_headroom,
                tail_margin=self.trim_tail_margin,
                min_cut=self.trim_min_cut,
            )
            for log in logs:
                logger.info(f"qwen_tts trim: {log}")
            if logs:
                logger.info(f"qwen_tts: trimmed WAV -> {out_file}")
        except Exception as e:
            logger.warning(f"qwen_tts: trim failed ({e}) - keeping original WAV")
