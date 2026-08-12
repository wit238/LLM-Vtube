"""JaiTTS (F5-TTS, Thai zero-shot voice cloning) TTS engine.

HTTP client for a local JaiTTS server (see the `jaitts_modal` project's
`server_local.py`). The server keeps the CUDA pipeline loaded; this engine
just POSTs the text + reference voice and saves the returned WAV.
"""

import time
from pathlib import Path

import httpx
from loguru import logger

from .tts_interface import TTSInterface

_DEFAULT_API_URL = "http://127.0.0.1:8021/synthesize"


class TTSEngine(TTSInterface):
    """JaiTTS/F5-TTS Thai voice cloning via a local HTTP server."""

    def __init__(
        self,
        api_url: str = _DEFAULT_API_URL,
        ref_audio_path: str = "",
        ref_text: str = "",
        speed: float = 1.0,
        seed: int = -1,
        nfe_step: int = 24,
        timeout: float = 300.0,
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
        self.api_url = api_url
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.speed = float(speed)
        self.seed = int(seed)
        self.nfe_step = int(nfe_step)
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
                logger.warning(
                    f"jaitts_tts: fallback TTS enabled -> {self.fallback_tts}"
                )
            else:
                logger.warning(
                    f"jaitts_tts: unknown fallback_tts '{self.fallback_tts}' "
                    f"(supported: 'edge_tts')"
                )
        return self._fallback_engine

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        if not text or not text.strip():
            raise ValueError("Empty text for JaiTTS synthesis")

        out_file = self.generate_cache_file_name(file_name_no_ext, "wav")

        data = {
            "text": text,
            "ref_text": self.ref_text,
            "speed": str(self.speed),
            "seed": str(self.seed),
            "nfe_step": str(self.nfe_step),
        }
        files = None
        if self.ref_audio_path:
            ref = Path(self.ref_audio_path)
            if not ref.is_file():
                raise FileNotFoundError(
                    f"JaiTTS reference audio not found: {self.ref_audio_path}"
                )
            files = {
                "ref_audio": (ref.name, ref.open("rb"), "audio/wav"),
            }

        logger.info(f"jaitts_tts: POST {self.api_url} ({len(text)} chars)")
        t0 = time.time()
        try:
            try:
                resp = httpx.post(
                    self.api_url, data=data, files=files, timeout=self.timeout
                )
            except httpx.HTTPError as e:
                raise ConnectionError(
                    f"JaiTTS server unreachable at {self.api_url}. "
                    "Deploy it with: cd jaitts_modal && uv run modal deploy main.py"
                ) from e

            resp.raise_for_status()
            if len(resp.content) == 0:
                raise RuntimeError("JaiTTS server returned an empty response")

            # Modal web endpoints report whether the function was cold-started
            # (X-Modal-Slot-Status: "cold start") or reused a warm container,
            # so we can surface it in the run_server terminal for monitoring.
            slot_status = resp.headers.get("x-modal-slot-status", "").strip()
            if slot_status:
                cold = "cold start" in slot_status.lower()
                logger.info(
                    f"jaitts_tts: Modal slot status: {slot_status}"
                    + (" (first call after idle - slower)" if cold else "")
                )

            with open(out_file, "wb") as f:
                f.write(resp.content)
            logger.info(
                f"jaitts_tts: got WAV in {time.time() - t0:.1f}s "
                f"({len(resp.content)} bytes) -> {out_file}"
            )

            if self.trim_audio:
                self._trim_wav(out_file, text)

            return out_file
        except (ConnectionError, httpx.HTTPError, RuntimeError) as e:
            # JaiTTS unavailable (not deployed / cold start / timeout) - fall
            # back to the configured engine (e.g. edge_tts) if one is set.
            logger.warning(f"jaitts_tts: {e}")
            engine = self._get_fallback_engine()
            if engine is not None:
                logger.warning(
                    f"jaitts_tts: falling back to {self.fallback_tts} "
                    f"(total so far {time.time() - t0:.1f}s)"
                )
                try:
                    result = engine.generate_audio(text, file_name_no_ext)
                except Exception as fe:
                    logger.error(f"jaitts_tts: fallback TTS also failed: {fe}")
                else:
                    if result:
                        logger.info(
                            f"jaitts_tts: fallback {self.fallback_tts} produced "
                            f"{result} in {time.time() - t0:.1f}s"
                        )
                        return result
                    logger.error(
                        "jaitts_tts: fallback TTS returned no audio - re-raising"
                    )
            raise
        finally:
            if files is not None:
                files["ref_audio"][1].close()

    def _trim_wav(self, out_file: str, text: str) -> None:
        """Post-process the generated WAV: cut leading garbage / trailing
        noise with faster-whisper word timestamps (same method as the
        jaitts_modal project's trim_leading.py)."""
        try:
            from ..utils.audio_trim import trim_wav_file, read_wav
        except Exception as e:
            logger.warning(f"jaitts_tts: audio_trim unavailable, skipping trim: {e}")
            return
        try:
            from pathlib import Path as _Path

            out = _Path(out_file)
            data, sr = read_wav(out)
            if len(data) / sr < 0.5:
                logger.info("jaitts_tts: clip too short for trim, skipping")
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
                logger.info(f"jaitts_tts trim: {log}")
            if logs:
                logger.info(f"jaitts_tts: trimmed WAV -> {out_file}")
        except Exception as e:
            logger.warning(f"jaitts_tts: trim failed ({e}) - keeping original WAV")
