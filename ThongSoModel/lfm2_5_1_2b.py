"""LiquidAI LFM2.5 1.2B Instruct Q4_K_M for an RTX 3050 6 GB."""

MODEL_NAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
MODEL_PATHS = (
    r"models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    r"R:\DoAn\AI\TreinKhaiWebAPIModel\Qwen2.5-1.5B\models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
)

# Llama server settings
LLAMA_THREADS = 30
LLAMA_NGL = 99
LLAMA_PARALLEL = 4
LLAMA_SLOT_CTX = 8192
LLAMA_BATCH = 512
LLAMA_UBATCH = 512

# Inference parameters
TEMPERATURE = 0.0
MAX_TOKENS = 128
TOP_P = 1.0
RETRY_COUNT = 3
