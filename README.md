# WebApiSystemPromptVector

Web API phân loại prompt sinh viên học code theo mô hình **rubric-grounded vector classification**:

```text
STUDENT_PROMPT + optional AI_OUTPUT
-> rule-based extractor bắt các dấu hiệu rõ ràng
-> LLM extractor gán vector semantic 44 chiều
-> merge rule vector + LLM vector
-> APC scoring engine chọn candidate level từ prompt
-> output alignment/mismatch engine quyết định giữ level hay hỏi lại sinh viên
```

Nguyên tắc mới:

- **Rubric là nền lý thuyết**: định nghĩa L0-L6 và 3 chiều Prompt: Artifact / Expectation / Contribution.
- **Vector là engine vận hành**: biến rubric thành feature có thể tính, debug, benchmark và chống bypass tốt hơn.
- **Prompt quyết định candidate level**.
- **AI_OUTPUT không tự nâng/hạ level sinh viên**. Output chỉ kiểm tra transaction có hợp lệ không.
- Nếu AI_OUTPUT khớp expectation -> giữ candidate level.
- Nếu AI_OUTPUT lệch expectation -> trả `level = "Can hoi lai SV"`, giữ `candidate_level`, `accept = 0`, `requires_student_confirmation = true`.

## Cấu trúc chính

```text
.
├── app.py                         # FastAPI endpoints: /classify, /classify_debug, /rubric, /health, /info
├── rubric_schema.py               # Rubric lý thuyết L0-L6 + 3 chiều Prompt + 6 chiều Output
├── rule_based_extractor.py        # Regex/rule extractor cho feature dễ nhận diện
├── classifier.py                  # Gọi llama-server, parse JSON/vector, merge rule + LLM vector
├── apc_v4_engine.py               # Công thức APC v4 + alignment_score + mismatch_score
├── vector_schema.py               # Tên và mô tả p1/p2/a1...r10/o1...m5 cho debug
├── config.py                      # Load .env và model profile
├── llama_manager.py               # Start/stop llama-server.exe
├── main.py                        # Entry point chạy API
├── prompts/
│   └── system_prompt_vector_v1.md # System prompt bắt model xuất 44-bit transaction vector
├── ThongSoModel/                  # Các profile model GGUF
└── benchmark_vector_features.py   # Benchmark/debug từng vector bằng /classify_debug
```

## Tại sao có cả rubric và vector?

Không chọn “rubric hoặc vector” theo kiểu một mất một còn. Hệ thống dùng:

```text
Rubric = theoretical framework / validation reference
Vector = operational layer / automatic scoring engine
```

Rubric giúp giải thích học thuật: mỗi level là gì, dựa trên Artifact / Expectation / Contribution. Vector giúp chạy nhanh và nhất quán: model không tự phán level, mà chỉ đánh dấu feature; engine deterministic mới tính level.

## Vector 44 bit

Thứ tự vector:

```text
p1, p2,
a1, a2, a3, a4, a5, a6, a7,
d1, d2, d3, d4, d5, d6, d7,
r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13,
m1, m2, m3, m4, m5
```

Trong đó:

- `p1 = is_coding`
- `p2 = has_context`
- `a1..a7 = Artifact`: sinh viên cung cấp gì trong prompt
- `d1..d7 = Contribution/Design`: sinh viên đã đóng góp gì về thuật toán, logic, cấu trúc, test, giả thuyết
- `r1..r10 = Expectation/Request`: sinh viên kỳ vọng AI làm vai trò gì
- `o1..o13 = Output`: AI_OUTPUT có mặt không, có đúng role/scope/agency/form/pedagogy không, và output thuộc dạng nào
- `m1..m5 = Mismatch`: output rộng quá, thiếu trả lời, nâng vai trò, làm thay sinh viên, sai form/sai pedagogy

## Prompt scoring

26 bit đầu xác định `candidate_level`:

```text
candidate_level = argmax S_k(prompt_vector), sau gating G_k
```

Các level:

- `L0`: ngoài phạm vi code hoặc không đủ transaction học code.
- `L1`: Full Delegation, SV giao AI tạo lời giải hoàn chỉnh.
- `L2`: Localized Delegation, SV có artifact/local context nhưng vẫn nhờ AI sửa/thêm/refactor/convert.
- `L3`: Exploratory Understanding, SV muốn hiểu nguyên nhân/cơ chế.
- `L4`: Guided Construction, SV có thiết kế/algorithm/formula/flow và AI chỉ implement theo.
- `L5`: Reflective Verification, SV đã có solution và AI review/check/test.
- `L6`: Minimal Augmentation, SV hỏi syntax/API/command/reference rất hẹp.

## Output alignment scoring

18 bit cuối không đổi năng lực sinh viên. Chúng tạo hai điểm:

```text
alignment_score = 0.25*role + 0.25*scope + 0.20*agency + 0.15*form + 0.15*pedagogy
mismatch_score  = 0.25*over_scope + 0.15*under_answer + 0.20*role_escalation + 0.25*agency_takeover + 0.15*form_pedagogy_mismatch
```

Mặc định:

```text
alignment_score >= 0.75 và mismatch_score < 0.30 và không có active mismatch -> output_aligned
ngược lại -> output_mismatch_requires_confirmation
```

Response luôn giữ lại:

- `candidate_level`: level từ prompt.
- `level`: kết quả cuối cùng; có thể là `Can hoi lai SV` nếu output mismatch.
- `final_status`: trạng thái cuối.
- `confirmation_reasons`: lý do cần hỏi lại SV.
- `alignment_score`, `mismatch_score`.

## Rule-based extractor

`rule_based_extractor.py` bắt các feature rõ ràng bằng rule/regex:

- code block / code-like syntax
- error/exception/stack trace
- SQL/schema/config/log
- từ khóa fix/debug/explain/review/test/syntax
- output complete solution / code patch / explanation / review / tests / narrow reference
- mismatch hiển nhiên như L6 syntax nhưng output full solution

Rule vector được merge với LLM vector bằng `max(rule, llm)`. Rule chỉ là **lower-bound override**: nếu regex thấy code/error/action rõ, feature đó phải bật; nhưng regex không tự tắt các phán đoán semantic của LLM.

Endpoint `/classify_debug` trả thêm:

- `llm_vector_detail`: vector LLM sinh ra.
- `final_vector_detail`: vector cuối đưa vào engine sau khi merge rule + LLM.
- `rule_evidence`: các rule hits cụ thể.

## API endpoints

- `POST /classify`: phân loại transaction.
- `POST /classify_debug`: phân loại + trả chi tiết vector.
- `GET /rubric`: xem rubric lý thuyết L0-L6 và các chiều đánh giá.
- `GET /health`: kiểm tra sức khỏe API/llama-server.
- `GET /info`: xem cấu hình.

## Request mẫu

```powershell
$body = @{
  prompt = "Cú pháp lệnh git để đổi tên branch hiện tại là gì?"
  ai_output = "Dùng git branch -m new-name để đổi tên branch hiện tại."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/classify_debug" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Response quan trọng:

```json
{
  "level": "L6",
  "candidate_level": "L6",
  "transaction_status": "output_aligned",
  "final_status": "accepted",
  "requires_student_confirmation": false,
  "alignment_score": 1.0,
  "mismatch_score": 0.0
}
```

Nếu output lệch:

```json
{
  "level": "Can hoi lai SV",
  "candidate_level": "L6",
  "transaction_status": "output_mismatch_requires_confirmation",
  "final_status": "requires_student_confirmation",
  "requires_student_confirmation": true,
  "confirmation_reasons": ["over_scope_broader", "role_escalation", "agency_takeover"]
}
```

## Cấu hình và chạy

Cài Python package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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

Nếu `llama-server.exe` không auto-detect được:

```env
LLAMA_SERVER_EXE=C:\path\to\llama-server.exe
```

Chạy API:

```powershell
python main.py
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Kiểm tra nhanh

```powershell
python -m py_compile app.py classifier.py apc_v4_engine.py vector_schema.py rubric_schema.py rule_based_extractor.py
python -m unittest -v
```

## Update v4 — Prompt A/E/C + Output 6D

Bản này đã được sửa theo 3 tài liệu/sheet mới:

- Prompt Candidate Level `Lp` được tính bằng 3 chiều: **Artifact / Expectation / Contribution**.
- Output Observed Level `Lo` được tính riêng bằng 6 chiều: **Role / Scope / Agency / Form / Pedagogy / Mismatch**.
- Nếu output lệch expectation, hệ thống **không đổi level của sinh viên**, giữ `CandidateLevel = Lp` và cảnh báo `WarningLevel = Lo`.
- L1 đã bao gồm: vibe code, mô tả nghiệp vụ bằng ngôn ngữ thường, AI làm từ đầu, thêm feature/module.
- L4 chỉ kích hoạt khi có pseudo code / flow code theo hướng hàm / HOW rõ ràng. Requirement dài hoặc vibe code không được tính là L4.
- L6 là tra cứu hẹp: lý thuyết ngắn, syntax/API, config, SQL, terminal, IDE. Nếu output có giải thích + ví dụ mở rộng thì `Lo` chuyển sang L3 và warning theo L3.

Các file chính đã sửa:

- `apc_v4_engine.py`: công thức vector mới, `Lp`, `Lo`, `MismatchScore`, `Fit`, `WarningLevel`.
- `vector_schema.py`: schema vector 60 chiều.
- `rule_based_extractor.py`: rule extractor mới theo rubric Prompt A/E/C + Output 6D.
- `rubric_schema.py`: định nghĩa lý thuyết L0-L6 mới.
- `prompts/system_prompt_vector_v1.md`: system prompt mới yêu cầu LLM xuất 60 feature.
- `app.py`: API response bổ sung `expected_output_level` và `warning_level`.

Kiểm thử nhanh:

```bash
python -m unittest discover -v
```

Case chuẩn:

```text
Prompt: What is printf in C?
Output: explanation + stdio.h + #include <stdio.h> + int main() + printf examples

CandidateLevel = L6
OutputObservedLevel = L3
WarningLevel = L3
Status = Provisional Warning hoặc Needs Student Confirmation
```
