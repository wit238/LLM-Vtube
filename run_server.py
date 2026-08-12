import os
import sys
import time
import atexit
import asyncio
import argparse
import subprocess
import threading
from pathlib import Path
import tomli
import urllib.parse
import urllib.request
import uvicorn
from loguru import logger
from upgrade_codes.upgrade_manager import UpgradeManager

from src.open_llm_vtuber.server import WebSocketServer
from src.open_llm_vtuber.config_manager import Config, read_yaml, validate_config

# Keeps the auto-started JaiTTS server process handle so it can be stopped on exit.
_jaitss_proc = None

os.environ["HF_HOME"] = str(Path(__file__).parent / "models")
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).parent / "models")

# Add virtual environment Scripts folder to PATH so ffmpeg can be found by pydub
venv_scripts = str(Path(sys.executable).parent)
if venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

import pydub

ffmpeg_exe = Path(venv_scripts) / "ffmpeg.exe"
if ffmpeg_exe.exists():
    pydub.AudioSegment.converter = str(ffmpeg_exe)

upgrade_manager = UpgradeManager()


def get_version() -> str:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


def _url_base(url: str) -> str:
    """Return scheme://netloc for a URL (used to derive a base for /health)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except BaseException:
        return False


def _detect_jaitts_device(venv_python: Path) -> str:
    """Ask the JaiTTS venv which torch device it can use.

    Runs a tiny probe inside the venv so we report the real torch build
    (CUDA vs CPU wheels) and whether a GPU is actually present, instead of
    relying on the machine's global torch.
    """
    probe = (
        "import torch;"
        "print('CUDA_AVAILABLE' if torch.cuda.is_available() else 'CPU');"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if result.returncode != 0 or not lines:
            logger.warning(
                f"JaiTTS device probe failed (rc={result.returncode}): "
                f"{result.stderr.strip()[:200]}"
            )
            return "cpu"
        device = "cuda" if lines[0] == "CUDA_AVAILABLE" else "cpu"
        logger.info(
            f"JaiTTS torch device: {device}"
            + (f" ({lines[1]})" if device == "cuda" and len(lines) > 1 else "")
        )
        return device
    except subprocess.TimeoutExpired:
        logger.warning("JaiTTS device probe timed out - assuming CPU.")
        return "cpu"
    except Exception as e:
        logger.warning(f"JaiTTS device probe error: {e} - assuming CPU.")
        return "cpu"


def ensure_jaitts_server(config: Config) -> None:
    """Auto-start the local JaiTTS server when jaitts_tts is selected.

    Uses the jaitts_tools project's own CUDA venv (a separate Python env).
    The server dir is taken from `server_dir`, or derived from the parent of
    `ref_audio_path`. Skips silently if the server is already healthy.

    The device (CUDA if a GPU is present, else CPU) is detected by probing the
    JaiTTS venv itself, and the chosen device is exported as an env var so
    `server_local.py` / `run_local.py` pick it up explicitly.
    """
    global _jaitss_proc
    try:
        tts = config.character_config.tts_config
        if tts.tts_model != "jaitts_tts":
            return
        jcfg = tts.jaitts_tts
        if jcfg is None or not getattr(jcfg, "auto_start", True):
            return
    except Exception as e:
        logger.warning(f"JaiTTS auto-start skipped (config): {e}")
        return

    health_url = _url_base(jcfg.api_url) + "/health"
    if _health_ok(health_url):
        logger.info("JaiTTS server already running - skipping auto-start.")
        return

    def _abs(path: Path) -> Path:
        """Resolve relative paths against the run_server.py directory so
        spawning the JaiTTS server (cwd=server_dir) never double-resolves."""
        return path if path.is_absolute() else Path(__file__).resolve().parent / path

    server_dir = (
        _abs(Path(os.path.expandvars(jcfg.server_dir)).expanduser())
        if jcfg.server_dir
        else Path("")
    )
    if not server_dir.is_dir():
        ref_parent = _abs(
            Path(os.path.expandvars(jcfg.ref_audio_path)).expanduser().parent
        )
        for candidate in (ref_parent, ref_parent / "jaitts_tools"):
            if (candidate / "server_local.py").exists():
                server_dir = candidate
                break
    venv_python = server_dir / ".venv" / "Scripts" / "python.exe"
    server_script = server_dir / "server_local.py"

    if (
        not server_dir.is_dir()
        or not venv_python.is_file()
        or not server_script.is_file()
    ):
        logger.error(
            f"Cannot auto-start JaiTTS: server_dir='{server_dir}' has no "
            f"server_local.py / .venv\\Scripts\\python.exe. "
            f"Set 'server_dir' in conf.yaml (jaitts_tts) or start it manually."
        )
        return

    device = _detect_jaitts_device(venv_python)

    logger.info(
        f"Auto-starting JaiTTS server ({device} venv) from {server_dir} ... "
        f"(first load may take a while)"
    )
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    jaitts_log = open(log_dir / "jaitts_server.log", "a", encoding="utf-8", buffering=1)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    # JaiTTS server must use the default HF cache (model already lives there),
    # NOT the workspace HF_HOME override inherited from run_server.py.
    child_env = os.environ.copy()
    child_env.pop("HF_HOME", None)
    child_env.pop("HF_HUB_CACHE", None)
    child_env.pop("MODELSCOPE_CACHE", None)
    # Explicitly tell the JaiTTS server which torch device to use.
    child_env["JAITTS_DEVICE"] = device
    try:
        _jaitss_proc = subprocess.Popen(
            [str(venv_python), "-X", "utf8", str(server_script)],
            cwd=str(server_dir),
            env=child_env,
            stdout=jaitts_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
    except Exception as e:
        jaitts_log.close()
        logger.error(f"Failed to spawn JaiTTS server: {e}")
        return

    deadline = time.monotonic() + jcfg.server_startup_timeout
    while time.monotonic() < deadline:
        if _jaitss_proc.poll() is not None:
            logger.error(
                f"JaiTTS server exited early with code {_jaitss_proc.returncode}."
            )
            _jaitss_proc = None
            return
        if _health_ok(health_url):
            logger.info(f"JaiTTS server is ready ({device} voice clone active).")
            return
        time.sleep(2)

    logger.warning(
        "JaiTTS server did not become ready in time. "
        "TTS requests may fail until it comes up."
    )


def stop_jaitts_server() -> None:
    global _jaitss_proc
    if _jaitss_proc is not None and _jaitss_proc.poll() is None:
        logger.info("Stopping auto-started JaiTTS server ...")
        _jaitss_proc.terminate()
        try:
            _jaitss_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _jaitss_proc.kill()
        _jaitss_proc = None


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


def check_frontend_submodule(lang=None):
    """
    Check if the frontend submodule is initialized. If not, attempt to initialize it.
    If initialization fails, log an error message.
    """
    if lang is None:
        lang = upgrade_manager.lang

    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        if lang == "zh":
            logger.warning("未找到前端子模块，正在尝试初始化子模块...")
        else:
            logger.warning(
                "Frontend submodule not found, attempting to initialize submodules..."
            )

        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"], check=True
            )
            if frontend_path.exists():
                if lang == "zh":
                    logger.info("👍 前端子模块（和其他子模块）初始化成功。")
                else:
                    logger.info(
                        "👍 Frontend submodule (and other submodules) initialized successfully."
                    )
            else:
                if lang == "zh":
                    logger.critical(
                        '子模块初始化失败。\n你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。'
                    )
                    logger.error(
                        "初始化子模块后，前端文件仍然缺失。\n"
                        + "你是否手动更改或删除了 `frontend` 文件夹？\n"
                        + "它是一个 Git 子模块 - 你不应该直接修改它。\n"
                        + "如果你这样做了，请使用 `git restore frontend` 丢弃你的更改，然后再试一次。\n"
                    )
                else:
                    logger.critical(
                        'Failed to initialize submodules. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.'
                    )
                    logger.error(
                        "Frontend files are still missing after submodule initialization.\n"
                        + "Did you manually change or delete the `frontend` folder?  \n"
                        + "It's a Git submodule — you shouldn't modify it directly.  \n"
                        + "If you did, discard your changes with `git restore frontend`, then try again.\n"
                    )
        except Exception as e:
            if lang == "zh":
                logger.critical(
                    f'初始化子模块失败: {e}。\n怀疑你跟 GitHub 之间有网络问题。你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。\n'
                )
            else:
                logger.critical(
                    f'Failed to initialize submodules: {e}. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.\n'
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Open-LLM-VTuber Server")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--hf_mirror", action="store_true", help="Use Hugging Face mirror"
    )
    return parser.parse_args()


@logger.catch
def run(console_log_level: str):
    init_logger(console_log_level)
    logger.info(f"Open-LLM-VTuber, version v{get_version()}")

    # Get selected language
    lang = upgrade_manager.lang

    # Check if the frontend submodule is initialized
    check_frontend_submodule(lang)

    # Sync user config with default config
    try:
        upgrade_manager.sync_user_config()
    except Exception as e:
        logger.error(f"Error syncing user config: {e}")

    atexit.register(WebSocketServer.clean_cache)

    # Load configurations from yaml file
    config: Config = validate_config(read_yaml("conf.yaml"))
    server_config = config.system_config

    # Start the local JaiTTS server if jaitts_tts is selected (CUDA voice clone)
    atexit.register(stop_jaitts_server)

    # Initialize the WebSocket server (synchronous part). The constructor
    # mounts all routes/static assets, so the app can answer HTTP requests
    # (e.g. the Railway healthcheck on "/") before heavy init finishes.
    server = WebSocketServer(config=config)

    if server_config.enable_proxy:
        logger.info("Proxy mode enabled - /proxy-ws endpoint will be available")

    def _background_init() -> None:
        """Load slow components (JaiTTS, RAG index, service context) after the
        HTTP server is already listening, so the healthcheck passes early."""
        try:
            ensure_jaitts_server(config)
        except Exception as e:
            logger.error(f"JaiTTS auto-start failed: {e}")

        # Build the file knowledge base (RAG) index from the configured folder
        try:
            from src.open_llm_vtuber.knowledge.knowledge_base import KnowledgeBase
            from src.open_llm_vtuber.knowledge.base import set_knowledge_base

            kcfg = config.character_config.knowledge_config
            if kcfg is not None and kcfg.enabled:
                kb = KnowledgeBase(kcfg)
                kb.build_index()
            else:
                kb = None
            set_knowledge_base(kb)
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            set_knowledge_base(None)

        # Perform asynchronous initialization (loading context, etc.)
        logger.info("Initializing server context...")
        try:
            asyncio.run(server.initialize())
            logger.info("Server context initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize server context: {e}")
            setattr(server, "_init_error", e)
            server.default_context_cache._init_error = e

    threading.Thread(target=_background_init, daemon=True).start()

    # Run the Uvicorn server (binds the port immediately; heavy init runs in background)
    port = int(os.environ.get("PORT", server_config.port))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app=server.app,
        host=host,
        port=port,
        log_level=console_log_level.lower(),
    )


if __name__ == "__main__":
    args = parse_args()
    console_log_level = "DEBUG" if args.verbose else "INFO"
    if args.verbose:
        logger.info("Running in verbose mode")
    else:
        logger.info(
            "Running in standard mode. For detailed debug logs, use: uv run run_server.py --verbose"
        )
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    run(console_log_level=console_log_level)
