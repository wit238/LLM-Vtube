import os
import re
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from loguru import logger
from .asr_interface import ASRInterface

_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")

def _thai_ratio(text: str) -> float:
    """Proportion of Thai characters among non-whitespace characters (0-1)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    thai = sum(1 for c in chars if _THAI_CHAR_RE.match(c))
    return thai / len(chars)


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        model_name: str = "typhoon-ai/typhoon-asr-realtime",
        device: str = "cpu",
        force_thai: bool = True,
        thai_threshold: float = 0.3,
    ) -> None:
        logger.info(f"Initializing Typhoon ASR ({model_name}) on {device}...")
        self.model_name = model_name
        self.device = device
        self.force_thai = bool(force_thai)
        self.thai_threshold = float(thai_threshold)

        # Lazy import typhoon_asr
        try:
            from typhoon_asr import transcribe
            self.transcribe_fn = transcribe
        except ImportError:
            logger.error("typhoon-asr package not found. Please install it using `pip install typhoon-asr`")
            raise

    @staticmethod
    def _extract_text(res) -> str:
        """Extract the transcript string from various typhoon_asr result shapes."""
        if res is None:
            return ""
        if isinstance(res, str):
            return res.strip()
        if isinstance(res, dict) and "text" in res:
            return VoiceRecognition._extract_text(res["text"])
        if hasattr(res, "text"):
            val = getattr(res, "text")
            if isinstance(val, str):
                return val.strip()
            return VoiceRecognition._extract_text(val)
        if isinstance(res, (list, tuple)) and len(res) > 0:
            return VoiceRecognition._extract_text(res[0])
        return str(res).strip()

    def _apply_thai_lock(self, text: str) -> str:
        """If Thai locking is on, reject transcripts that are not predominantly
        Thai (e.g. English/other speech fed to the Thai-only model)."""
        if not self.force_thai:
            return text
        if not text:
            return text
        ratio = _thai_ratio(text)
        if ratio < self.thai_threshold:
            logger.warning(
                f"Typhoon ASR: non-Thai audio detected (Thai ratio {ratio:.0%} "
                f"< {self.thai_threshold:.0%}) - ignoring: {text[:60]!r}"
            )
            return ""
        return text

    def transcribe_np(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""

        # Make sure audio is 16-bit PCM for wavfile
        audio = np.clip(audio, -1, 1)
        audio_int16 = (audio * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            wavfile.write(tmp_path, self.SAMPLE_RATE, audio_int16)
            result = self.transcribe_fn(tmp_path, model_name=self.model_name, device=self.device)

            text = self._extract_text(result)
            return self._apply_thai_lock(text)
        except Exception as e:
            logger.error(f"Error during Typhoon ASR transcription: {e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
