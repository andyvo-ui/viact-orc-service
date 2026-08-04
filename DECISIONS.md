# Nhật ký quyết định — ocr-service

Ghi lại **vì sao** kiến trúc thành ra như hiện tại. Code và SETUP.md nói *cái gì*;
file này nói *tại sao*, và **điều kiện nào thì phải xem lại**.

---

## 1. Bối cảnh

- Phần cứng: Intel Core Ultra 9 285K · 47GB RAM · **RTX 5090 32GB VRAM (Blackwell, sm_120)**
- GPU đã có Qwen 27B chiếm ~18–24GB; tương lai thêm STT ~2–3GB
- Khách hàng ở **Hong Kong** → ngôn ngữ chính: **Traditional Chinese + English**
- Yêu cầu: OCR chạy **hoàn toàn local**, không gọi cloud AI, expose cho service khác dùng

## 2. Ràng buộc quyết định kiến trúc: sm_120

Đây là thứ định hình mọi lựa chọn còn lại, **không phải VRAM**.

RTX 5090 là Compute Capability 12.0. PaddlePaddle native là runtime tụt hậu nhất
trong nhóm: wheel `paddlepaddle-gpu` chính thức không ship kernel sm_120 (tính đến
12/2025 vẫn phải dùng community wheel build tay). Paddle build kernel cứng theo
arch nên không có đường JIT thoát.

**Nguyên tắc rút ra, áp dụng cho cả STT sau này:** ưu tiên runtime đã có sẵn kernel
sm_120 (vLLM, PyTorch cu128+, ONNX Runtime, TensorRT 10.x); tránh runtime tự build
kernel riêng (PaddlePaddle native, CTranslate2).

Cửa thoát: PaddleOCR 3.5+ tách engine khỏi framework (`engine=` chọn `paddle` /
`transformers` / `onnxruntime`). **PaddlePaddle chỉ bắt buộc khi train/export**,
không bắt buộc để inference.

## 3. Kiến trúc chốt

```
                  ocr-gateway :8000  (container)
                  ├── FAST LANE  — import in-process, KHÔNG phải service riêng
                  │     PP-OCRv6_small det+rec (ONNX) · CPU · ~31MB
                  │     POST /ocr → text + boxes + scores
                  │
                  └── DOC LANE   — HTTP ra ngoài compose, tới Ollama trên GPU host
                        (gateway/.env: DOC_LANE_URL, mặc định :11434)
                        PaddleOCR-VL-1.6-0.9B trên Ollama · GPU · không do repo này quản
                        POST /parse → Markdown/JSON giữ cấu trúc
```

**Đổi hướng (sau bản đầu):** doc lane từng chạy như container `paddleocr genai_server`
(vLLM) trong `docker-compose.yml`, cổng 8118. Đã bỏ — GPU host đã cài sẵn Ollama và
serve model đó trực tiếp, nên container vLLM trong repo này là dư thừa. `MODEL` và
`DOC_LANE_URL` giờ đọc từ `gateway/.env` (namespace/tag kiểu Ollama, không phải
đường dẫn model của vLLM).

## 4. Bảng quyết định

| # | Quyết định | Chọn | Vì sao | Xem lại khi |
|---|---|---|---|---|
| 1 | Model OCR | **PP-OCRv6_small** | gap so với medium trên đúng 2 ngôn ngữ cần: TC 77.0 vs 78.6, EN 93.3 vs 94.1 — nhỏ | benchmark HK thật không đạt → đổi 2 dòng `DET_MODEL`/`REC_MODEL` sang medium |
| 2 | Fast lane chạy ở đâu | **CPU** | GPU dùng chung với LLM → time-slicing làm p95 latency không đoán được. CPU cho latency ổn định, độc lập tải LLM. Đồng thời né sạch rủi ro sm_120 | đo p95 dưới tải thật thấy thiếu throughput |
| 3 | Doc lane | **PaddleOCR-VL-1.6** | đạt 96.3% OmniDocBench, cao hơn StructureV3. Ban đầu chọn vì chạy được trên vLLM (sm_120 sẵn); nay serve qua Ollama trên GPU host, ngoài repo này | — |
| 4 | Backend inference | **onnxruntime** | official ONNX weights có sẵn trên HF; không cần cài paddlepaddle | — |
| 5 | Gateway + fast lane | **chung 1 container** | gateway `import` trực tiếp `ocr_engine.py`, chạy in-process | muốn scale riêng fast lane → phải viết lại thành HTTP service |
| 6 | Weights fast lane | **bake vào image lúc build** | container chạy offline, request đầu không chờ tải, build fail = biết sớm | — |
| 7 | `depends_on` doc-lane | **KHÔNG** (nay không còn áp dụng — doc lane không phải service trong compose này nữa) | `/ocr` phải sống kể cả khi GPU lane chết. Trước đây còn tránh `up gateway` kéo theo image vLLM doc-lane (5.8GB); từ khi doc lane chuyển sang Ollama ngoài repo, service đó không còn tồn tại trong `docker-compose.yml` để mà depends_on | — |

## 5. Đã loại và lý do

| Loại | Vì sao |
|---|---|
| `paddlex --serve` + PP-StructureV3 trên paddle-GPU | đường mặc định mọi tutorial chỉ, nhưng đặt toàn bộ service lên runtime yếu nhất trên Blackwell. Phải build from source `CUDA_ARCH_BIN=120` hoặc dùng dev wheel cộng đồng |
| PaddlePaddle community sm_120 wheel | pin Python 3.10 + CUDA 13 + bản `.dev`, cập nhật lần cuối 12/2025 — nợ vận hành không đáng |
| High-Stability Serving (Triton) | giải bài toán multi-tenant/autoscale mà single-box không có |
| PP-OCRv6_tiny | thiếu Japanese (49/50 ngôn ngữ), rec accuracy chỉ 73.5% |
| PP-OCRv5_server | thế hệ cũ; v6_medium nhỏ hơn ~10× mà tốt hơn |
| Dùng Qwen-VL thay OCR để đọc chữ | VLM nén ảnh xuống lưới token cố định **trước khi** model nhìn thấy → chữ nhỏ bị phá huỷ. Với chữ Hán (phân biệt bằng nét nhỏ) VLM hay bịa ký tự. OCR cắt riêng dải chữ, đọc ở độ phân giải gốc |

## 6. Ranh giới năng lực — OCR không phải "hiểu ảnh"

| Nhu cầu | Tool |
|---|---|
| Đọc chữ biển báo / bảng tên / số hiệu thiết bị | PaddleOCR fast lane |
| PDF/hợp đồng có bảng → Markdown | PaddleOCR doc lane |
| "Công nhân có đội mũ không", đếm người, PPE | **Qwen-VL** (không thuộc service này) |
| Ảnh công trường có chữ quan trọng | **cả hai** — OCR lấy text chính xác, đưa vào prompt Qwen-VL làm context |

Model OCR không nhìn cả bức ảnh: det tìm box chữ, rec chỉ nhận dải ~48×320px đã
cắt. Đó là lý do 31MB đủ — và cũng là lý do nó **không bao giờ** hiểu được cảnh,
bất kể phóng to bao nhiêu.

## 7. Số tham chiếu

| | tiny | small | medium |
|---|---|---|---|
| Params | 1.5M | 7.7M | 34.5M |
| Traditional Chinese | — | 77.0% | 78.6% |
| Printed English | 77.3% | 93.3% | 94.1% |
| Printed Chinese | — | 90.5% | 91.5% |
| Rec accuracy (avg) | 73.5% | 81.3% | 83.2% |
| A100 (paddle) | 0.13s | 0.25s | 0.29s |
| Xeon + OpenVINO | — | 0.59s | 1.40s |
| Ngôn ngữ | 49 | 50 | 50 |

Version: thư viện `paddleocr` **3.7.0** (2026-06-11) là bản đầu tiên ship PP-OCRv6.
`>=3.5` hay `>=3.6` sẽ **không** resolve được tên model `PP-OCRv6_*`.

Ngân sách VRAM ước lượng: Qwen 20GB (phải cap cứng) + doc-lane 4–5GB + STT 2–3GB
≈ 27–32GB / 32GB. Sát trần — mọi service phải cap tường minh.

## 8. Chưa verify — xem cuối SETUP.md

Đã giải quyết bằng cách đọc source của wheel `paddleocr==3.7.0` + `paddlex==3.7.0`
(không cần container): tên kwarg `engine`/`engine_config` **đúng**, model
`PP-OCRv6_small_det/_rec` **có** bản ONNX chính thức, và PDF vào `/ocr` chạy được
(`pypdfium2` nằm trong extra `ocr-core`). Chi tiết ở SETUP.md.

Đã sửa 2 bug tìm ra khi review (xem §10).

Còn lại quan trọng nhất: **benchmark 30–50 tài liệu HK thật** (Traditional Chinese +
English). Đây là thứ quyết định small có đủ hay phải lên medium, và không ai làm
thay được.

## 9. Điểm chưa quyết

~~`use_textline_orientation`~~ → **đã quyết, xem §11.** Input thực tế là **~50/50**
ảnh hiện trường và tài liệu scan, nên không có một giá trị global nào đúng.

**`/ocr` chạy blocking trong `async def`** → request đồng thời bị serialise, và
`/health` có thể vượt timeout 5s của compose healthcheck khi có tải. Cố tình
**chưa sửa**: đây là lựa chọn thiết kế (`run_in_threadpool`, đổi sang `def`, hay
chấp nhận serialise + scale bằng nhiều replica), không phải bug. `test_c1`/`test_c2`
đo được con số thật để quyết.

## 10. Bug đã sửa sau review

| Bug | Hậu quả | Sửa ở đâu |
|---|---|---|
| `rec_polys` là `list[np.ndarray]`, trả thẳng cho FastAPI | **`/ocr` trả 500 với mọi ảnh có chữ.** Ảnh trắng trả 200 (list rỗng) nên `/health` và test ảnh trắng vẫn xanh → bug ẩn rất kỹ | `run_ocr` coerce về builtin trong `ocr_engine.py` |
| `raise_for_status()` nằm trong `except httpx.HTTPError`; `HTTPStatusError` là subclass của `HTTPError` | doc lane trả 400 ("file của bạn sai") → caller nhận 502 ("GPU box chết") → retry vô hạn một request không bao giờ thành công | `/parse` tách 3 nhánh: transport → 502, upstream 4xx → pass through, upstream 5xx → 502 |
| Fixture `doc_lane_up` suy ra trạng thái doc lane từ status code của gateway | Do bug trên, doc lane **sống** mà từ chối file rác cũng ra 502 → fixture báo "chết". Toàn bộ nhóm `doclane_down` **pass mà không hề có điều kiện của nó** → claim §7 chưa từng được kiểm | probe trực tiếp cổng doc lane (khi đó 8118/vLLM, nay 11434/Ollama), không qua gateway |
| `check_device.py` trỏ vào `fast_lane/models/...` | Đường dẫn không bao giờ tồn tại (paddlex cache ở `~/.paddlex/official_models/`) → script luôn FAIL sai lý do, mà đây là công cụ duy nhất trả lời câu hỏi sm_120 | glob trong cache thật của paddlex |
| `predict()` yield `{"error": ...}` rồi **chạy tiếp** (không `return`) | `.get("rec_texts", [])` biến record lỗi đó thành **một page rỗng** → số page nhiều hơn thật đúng 1, trông y như tài liệu có trang trắng | `run_ocr` raise `EngineRejectedInput` |
| `/ocr` không có giới hạn size lẫn số page | `await file.read()` nạp cả file vào RAM; PDF 200 page block event loop ~2 phút → `/health` chết → compose restart container đang chạy tốt | stream ra đĩa + cap `OCR_MAX_UPLOAD_BYTES` / `OCR_MAX_PAGES`, trả 413 |

## 11. `/ocr` nhận PDF: quyết định, không phải tai nạn

Trước đây `/ocr` nhận PDF **do tình cờ**: `paddleocr` kéo theo `pypdfium2` (extra
`ocr-core`), nên `predict()` đọc được PDF và `run_ocr` loop theo page. Docstring thì
ghi "Single image", DECISIONS §6 ghi PDF → doc lane. Test lại assert PDF chạy được.
Code, doc và test nói ba chuyện khác nhau.

**Ranh giới thật không phải image vs PDF, mà là FLAT TEXT vs CÓ CẤU TRÚC:**

| input | lane | vì sao |
|---|---|---|
| Ảnh (biển báo, bảng tên, số hiệu) | `/ocr` | đúng mục đích ban đầu |
| PDF **scan** ngắn, chỉ cần text | `/ocr` | PDF scan = ảnh bọc trong PDF, cùng một việc |
| PDF có **bảng**, hợp đồng dài | `/parse` | `/ocr` vẫn trả 200 nhưng **âm thầm bỏ cấu trúc** |

Số page là proxy rẻ cho ranh giới đó → cap `OCR_MAX_PAGES=10` (mặc định), quá thì
**413 kèm câu chỉ sang `/parse`**. Không chặn PDF hẳn: mất đường chạy không cần GPU
cho một bản scan 2 trang là mất thật.

Cap 10 là **tạm**, bị ràng bởi chuyện blocking ở §9 — không phải giới hạn của model.
Sau khi chạy `test_c1`/`test_c2` trên máy thật thì nâng.

## 12. Hai pipeline cho hai loại input

Input ~50/50 ảnh hiện trường + tài liệu scan → một giá trị `use_textline_orientation`
global thì **sai một nửa số request**. Nên `/ocr?orientation=auto|upright`, mỗi setting
một pipeline riêng.

**Vì sao phải hai pipeline chứ không truyền cờ theo từng `predict()`** — đã xác minh
trong source paddlex 3.7.0: `OCRPipeline.__init__` chỉ tạo model textline-orientation
khi cờ bật **lúc construct** (`pipelines/ocr/pipeline.py:89-96`). Truyền
`use_textline_orientation=True` vào `predict()` của pipeline dựng với `False` thì
`check_model_settings_valid()` log lỗi và `predict()` **yield `{"error": ...}` rồi chạy
tiếp** — tức là thêm một page rỗng, không phải bật được orientation.

Giá phải trả: thêm 1 model ~7MB (`PP-LCNet_x1_0_textline_ori`, **có bản ONNX chính
thức** nên không kéo paddlepaddle vào — §4 vẫn đúng) và một bộ ONNX session nữa.
`warmup()` dựng **cả hai** lúc boot vì container chạy offline (§6).

**Mặc định = `auto`.** Hai kiểu sai không đối xứng: giả định upright cho ảnh chụp chéo
thì **mất chữ âm thầm và trông y như thành công**; bật orientation cho bản scan phẳng
thì chỉ tốn thêm chút latency. Chọn cái sai rẻ hơn. Đo xong muốn đổi thì
`OCR_DETECT_ORIENTATION=0` — và sửa `test_default_is_orientation_detection` kèm lý do.
