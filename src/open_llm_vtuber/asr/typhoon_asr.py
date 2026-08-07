import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from loguru import logger
from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        model_name: str = "typhoon-ai/typhoon-asr-realtime",
        device: str = "cpu",
    ) -> None:
        logger.info(f"Initializing Typhoon ASR ({model_name}) on {device}...")
        self.model_name = model_name
        self.device = device
        
        # Lazy import typhoon_asr
        try:
            from typhoon_asr import transcribe
            self.transcribe_fn = transcribe
        except ImportError:
            logger.error("typhoon-asr package not found. Please install it using `pip install typhoon-asr`")
            raise

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
            
            def extract_text(res):
                if res is None:
                    return ""
                if isinstance(res, str):
                    return res.strip()
                if hasattr(res, "text"):
                    return str(getattr(res, "text")).strip()
                if isinstance(res, dict) and "text" in res:
                    return str(res["text"]).strip()
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    return extract_text(res[0])
                return str(res).strip()

            return extract_text(result)
        except Exception as e:
            logger.error(f"Error during Typhoon ASR transcription: {e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
