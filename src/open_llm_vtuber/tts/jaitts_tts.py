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
        timeout: float = 300.0,
        trim_audio: bool = True,
        trim_model: str = "small",
        trim_device: str = "cpu",
        trim_compute_type: str = "int8",
        trim_download_root: str = "models/whisper",
        trim_headroom: float = 0.10,
        trim_tail_margin: float = 0.15,
        trim_min_cut: float = 0.30,
    ) -> None:
        self.api_url = api_url
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.speed = float(speed)
        self.seed = int(seed)
        self.timeout = float(timeout)
        self.trim_audio = bool(trim_audio)
        self.trim_model = trim_model
        self.trim_device = trim_device
        self.trim_compute_type = trim_compute_type
        self.trim_download_root = trim_download_root
        self.trim_headroom = float(trim_headroom)
        self.trim_tail_margin = float(trim_tail_margin)
        self.trim_min_cut = float(trim_min_cut)

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        if not text or not text.strip():
            raise ValueError("Empty text for JaiTTS synthesis")

        out_file = self.generate_cache_file_name(file_name_no_ext, "wav")

        data = {
            "text": text,
            "ref_text": self.ref_text,
            "speed": str(self.speed),
            "seed": str(self.seed),
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
            resp = httpx.post(
                self.api_url, data=data, files=files, timeout=self.timeout
            )
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"JaiTTS server unreachable at {self.api_url}. "
                "Deploy it with: cd jaitts_modal && uv run modal deploy main.py"
            ) from e
        finally:
            if files is not None:
                files["ref_audio"][1].close()

        resp.raise_for_status()
        if len(resp.content) == 0:
            raise RuntimeError("JaiTTS server returned an empty response")

        with open(out_file, "wb") as f:
            f.write(resp.content)
        logger.info(
            f"jaitts_tts: got WAV in {time.time() - t0:.1f}s "
            f"({len(resp.content)} bytes) -> {out_file}"
        )

        if self.trim_audio:
            self._trim_wav(out_file, text)

        return out_file

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
