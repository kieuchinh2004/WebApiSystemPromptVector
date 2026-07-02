import importlib.util
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# ─── API Server ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─── llama-server backend ────────────────────────────────────────────────────
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "28284"))
LLAMA_HEALTH_URL = f"http://127.0.0.1:{LLAMA_PORT}/health"
LLAMA_CHAT_URL   = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"

# Path to script directory
_here = os.path.dirname(os.path.abspath(__file__))

def _first_existing(paths):
    return next((p for p in paths if p and os.path.exists(p)), None)

# Path to llama-server.exe. Precedence: .env, then system programs.
LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE") or _first_existing([
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs", "Ollama", "lib", "ollama", "llama-server.exe"),
    os.path.join(os.getenv("ProgramFiles", ""), "Ollama", "lib", "ollama", "llama-server.exe"),
    os.path.join(os.getenv("ProgramFiles(x86)", ""), "Ollama", "lib", "ollama", "llama-server.exe"),
    r"C:\Users\acer\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe",
]) or r"C:\Users\acer\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"

# Model configuration profile from ThongSoModel
MODEL_CONFIG_FILE = os.getenv("MODEL_CONFIG_FILE", "qwen2_5_3b.py")

def _load_model_config(file_name):
    config_path = os.path.join(_here, "ThongSoModel", file_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Không tìm thấy file thông số model: {config_path}")

    spec = importlib.util.spec_from_file_location("selected_model_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp file thông số model: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_model_config = _load_model_config(MODEL_CONFIG_FILE)
MODEL_NAME = _model_config.MODEL_NAME

# Convert relative model paths from model config profile to absolute
_model_paths = [
    path if os.path.isabs(path) else os.path.join(_here, path)
    for path in _model_config.MODEL_PATHS
]
MODEL_PATH = os.getenv("MODEL_PATH") or _first_existing(_model_paths)

# ─── System Prompt File ──────────────────────────────────────────────────────
PROMPT_FILE = os.getenv("PROMPT_FILE") or _first_existing([
    os.path.join(_here, "prompts", "system_prompt_vector_v1.md"),
])

# Llama server settings with overrides from .env
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", str(getattr(_model_config, "LLAMA_THREADS", 8))))
LLAMA_NGL = int(os.getenv("LLAMA_NGL", str(getattr(_model_config, "LLAMA_NGL", 99))))
LLAMA_PARALLEL = int(os.getenv("LLAMA_PARALLEL", str(getattr(_model_config, "LLAMA_PARALLEL", 4))))
LLAMA_SLOT_CTX = int(os.getenv("LLAMA_SLOT_CTX", str(getattr(_model_config, "LLAMA_SLOT_CTX", 5632))))
LLAMA_CONTEXT = LLAMA_PARALLEL * LLAMA_SLOT_CTX
LLAMA_BATCH = int(os.getenv("LLAMA_BATCH", str(getattr(_model_config, "LLAMA_BATCH", 512))))
LLAMA_UBATCH = int(os.getenv("LLAMA_UBATCH", str(getattr(_model_config, "LLAMA_UBATCH", 512))))

# Inference tuning parameters (loaded from ThongSoModel profile, falling back to defaults)
TEMPERATURE = float(os.getenv("TEMPERATURE", str(getattr(_model_config, "TEMPERATURE", 0.0))))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(getattr(_model_config, "MAX_TOKENS", 256))))
TOP_P = float(os.getenv("TOP_P", str(getattr(_model_config, "TOP_P", 1.0))))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", str(getattr(_model_config, "RETRY_COUNT", 3))))
