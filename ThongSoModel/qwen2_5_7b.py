"""Thông số Qwen2.5 7B Q4 cho RTX 3050 6 GB."""

MODEL_NAME = "prompt-classifier-7b-q4.gguf"
MODEL_PATHS = (
    MODEL_NAME,
    r"R:\DoAn\AI\TrienKhaiExeModel\Qwen7b_Systempromt_APC1.3\prompt-classifier-7b-q4.gguf",
)

# Llama server settings
LLAMA_THREADS = 8
LLAMA_NGL = 99
LLAMA_PARALLEL = 4
LLAMA_SLOT_CTX = 5632
LLAMA_BATCH = 512
LLAMA_UBATCH = 512

# Inference parameters
TEMPERATURE = 0.0
MAX_TOKENS = 128
TOP_P = 1.0
RETRY_COUNT = 3
