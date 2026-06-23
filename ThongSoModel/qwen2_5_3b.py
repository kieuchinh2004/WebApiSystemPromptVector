"""Thông số Qwen2.5 3B Instruct Q4_K_M cho RTX 3050 6 GB."""

MODEL_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATHS = (
    MODEL_NAME,
    r"R:\DoAn\AI\TreinKhaiWebAPIModel\Qwen2.5-1.5B\qwen2.5-3b-instruct-q4_k_m.gguf",
)

# Llama server settings
LLAMA_THREADS = 15
LLAMA_NGL = 99
LLAMA_PARALLEL = 6
LLAMA_SLOT_CTX = 5632
LLAMA_BATCH = 512
LLAMA_UBATCH = 512

# Inference parameters
TEMPERATURE = 0.0
MAX_TOKENS = 128
TOP_P = 1.0
RETRY_COUNT = 3
