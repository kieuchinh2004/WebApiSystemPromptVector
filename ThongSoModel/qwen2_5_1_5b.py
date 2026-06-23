"""Thông số Qwen2.5 1.5B Q4 cho RTX 3050 6 GB."""

MODEL_NAME = "qwen2.5-1.5b-q4.gguf"
MODEL_PATHS = (
    MODEL_NAME,
    r"R:\DoAn\AI\TrienKhaiExeModel\Qwen2.5-1.5b_SystempromtVer2_APC1.3\qwen2.5-1.5b-q4.gguf",
    r"T:\Ollama\Model\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4",
)

# Llama server settings
LLAMA_THREADS = 30
LLAMA_NGL = 99
LLAMA_PARALLEL = 8
LLAMA_SLOT_CTX = 5632
LLAMA_BATCH = 512
LLAMA_UBATCH = 512

# Inference parameters
TEMPERATURE = 0.0
MAX_TOKENS = 128
TOP_P = 1.0
RETRY_COUNT = 3
