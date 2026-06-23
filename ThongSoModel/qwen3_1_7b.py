"""Qwen3 1.7B Q4_K_M (ggml-org conversion) for an RTX 3050 6 GB."""

MODEL_NAME = "Qwen3-1.7B-Q4_K_M.gguf"
MODEL_PATHS = (
    r"models\Qwen3-1.7B-Q4_K_M.gguf",
    r"R:\DoAn\AI\TreinKhaiWebAPIModel\Qwen2.5-1.5B\models\Qwen3-1.7B-Q4_K_M.gguf",
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
