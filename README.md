# WebApiSystemPromptVector

Web API phân loại prompt theo hướng:

```text
student prompt + system prompt
-> LLM sinh vector 26 bit
-> công thức APC v4 tính level / score / confidence
```

Pipeline hiện tại không dùng rule/keyword để sửa vector. Vector đưa vào công thức là vector do model sinh ra.

## Cấu trúc chính

```text
.
├── app.py                         # FastAPI endpoints: /classify, /classify_debug, /health, /info
├── classifier.py                  # Gọi llama-server, parse JSON/vector từ LLM
├── apc_v4_engine.py               # Công thức APC v4 tính score/level/confidence
├── vector_schema.py               # Tên và mô tả p1/p2/a1...r10 cho debug
├── config.py                      # Load .env và model profile
├── llama_manager.py               # Start/stop llama-server.exe
├── main.py                        # Entry point chạy API
├── prompts/
│   └── system_prompt_vector_v1.md # System prompt bắt model xuất vector
├── ThongSoModel/                  # Các profile model GGUF
└── benchmark_vector_features.py   # Benchmark/debug từng vector bằng /classify_debug
```

## Vector 26 bit

Thứ tự vector:

```text
p1, p2,
a1, a2, a3, a4, a5, a6, a7,
d1, d2, d3, d4, d5, d6, d7,
r1, r2, r3, r4, r5, r6, r7, r8, r9, r10
```

Trong đó:

- `p1 = is_coding`
- `p2 = has_context`
- `a1..a7 = artifact/context features`
- `d1..d7 = design/analysis features`
- `r1..r10 = request-type features`

Endpoint `/classify_debug` trả thêm:

- `llm_vector_detail`: vector model sinh ra.
- `final_vector_detail`: vector đưa vào công thức. Hiện tại bằng `llm_vector_detail`.

## Yêu cầu môi trường

- Windows
- Python 3.10+
- `llama-server.exe` của Ollama hoặc llama.cpp compatible server
- Model `.gguf`

Cài Python package:

```powershell
cd R:\DoAn\AI\TreinKhaiWebAPIModel\WebAPI_Vector_Check
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Hoặc chạy:

```powershell
SETUP_WINDOWS.bat
```

## Cấu hình model

Copy `.env.example` thành `.env`:

```powershell
copy .env.example .env
```

Các biến chính:

```env
API_HOST=0.0.0.0
API_PORT=8000
LLAMA_PORT=28284
MODEL_CONFIG_FILE=qwen2_5_1_5b.py
```

Nếu `llama-server.exe` không auto-detect được, set:

```env
LLAMA_SERVER_EXE=C:\path\to\llama-server.exe
```

Nếu model không nằm đúng path trong profile, override trực tiếp:

```env
MODEL_PATH=C:\path\to\your-model.gguf
```

## Model profiles

Các profile nằm trong `ThongSoModel/`.

| Profile | Expected GGUF name | Gợi ý vị trí đặt model |
|---|---|---|
| `qwen2_5_1_5b.py` | `qwen2.5-1.5b-q4.gguf` | đặt ngay cạnh source hoặc set `MODEL_PATH` |
| `qwen2_5_3b.py` | `qwen2.5-3b-instruct-q4_k_m.gguf` | đặt ngay cạnh source hoặc set `MODEL_PATH` |
| `qwen2_5_7b.py` | `prompt-classifier-7b-q4.gguf` | đặt ngay cạnh source hoặc set `MODEL_PATH` |
| `lfm2_5_1_2b.py` | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | `models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf` hoặc set `MODEL_PATH` |
| `qwen3_1_7b.py` | `Qwen3-1.7B-Q4_K_M.gguf` | `models\Qwen3-1.7B-Q4_K_M.gguf` hoặc set `MODEL_PATH` |
| `smollm2_1_7b.py` | `smollm2-1.7b-instruct-q4_k_m.gguf` | `models\smollm2-1.7b-instruct-q4_k_m.gguf` hoặc set `MODEL_PATH` |

Ví dụ dùng Qwen2.5 3B:

```env
MODEL_CONFIG_FILE=qwen2_5_3b.py
MODEL_PATH=C:\Models\qwen2.5-3b-instruct-q4_k_m.gguf
```

Ví dụ dùng SmolLM2:

```env
MODEL_CONFIG_FILE=smollm2_1_7b.py
MODEL_PATH=C:\Models\smollm2-1.7b-instruct-q4_k_m.gguf
```

Lưu ý: repository không chứa file `.gguf`. Không commit model vào git.

## Chạy API

```powershell
cd R:\DoAn\AI\TreinKhaiWebAPIModel\WebAPI_Vector_Check
python main.py
```

Hoặc:

```powershell
RUN_API.bat
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health:

```text
GET http://127.0.0.1:8000/health
```

## Test vector debug

Request:

```powershell
$body = @{
  prompt = "Explain this C++ snippet: int x = 1;"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/classify_debug" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Response sẽ có:

```json
{
  "level": "L3",
  "vector": "11100000100000000000010000",
  "llm_vector_detail": [],
  "final_vector_detail": []
}
```

## Benchmark vector

```powershell
$env:DATASET="R:\DoAn\AI\Dataset\Test_folder\apc_v13_synthetic_100_20260623.csv"
python benchmark_vector_features.py
```

Nếu CSV có cột đáp án vector như `p1,p2,a1...r10` hoặc `true_a1...true_r10`, script sẽ tính accuracy từng vector.

## Kiểm tra nhanh

```powershell
python -m py_compile app.py classifier.py apc_v4_engine.py vector_schema.py
python -m unittest -v
```
