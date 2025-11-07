# RAG-Based Question Answering System for Online PDF Documents

## 1. Giới thiệu tổng quan
Dự án triển khai hệ thống hỏi–đáp theo kiến trúc Retrieval-Augmented Generation (RAG) dành cho các tài liệu học thuật tiếng Việt trích xuất từ Instruct2DS Dataset. Toàn bộ pipeline được điều phối bằng **LangGraph** với mô hình điều khiển chính là `gemini-2.0-flash-thinking-exp-1219`. Mục tiêu của README này là cung cấp hướng dẫn chi tiết nhất có thể để bạn có thể:

1. Chuẩn bị môi trường và cấu hình cần thiết.
2. Nạp và quản lý dữ liệu vào cơ sở tri thức vector hóa.
3. Thực thi pipeline truy vấn để nhận câu trả lời kèm trích dẫn.

## 2. Kiến trúc hệ thống
Hệ thống gồm hai pha chính, mỗi pha được mô hình hóa dưới dạng một LangGraph riêng:

| Pha | Graph | Chức năng chính | Tập tin liên quan |
| --- | --- | --- | --- |
| Data Ingestion & Management | `build_ingestion_graph` | Đọc dữ liệu → Tiền xử lý → Chia đoạn → Tạo embedding → Lưu ChromaDB | `src/graphs/ingestion_graph.py`, `src/nodes/ingestion_nodes.py` |
| Query & Answer Generation | `build_query_graph` | Nhận câu hỏi → Truy xuất vector → Gọi Gemini → Hậu xử lý câu trả lời | `src/graphs/query_graph.py`, `src/nodes/query_nodes.py` |

Các cấu hình dùng chung được quản lý trong `src/config/settings.py`. Entry point duy nhất để chạy toàn bộ graph là `main.py`.

> ⚠️ **Lưu ý:** Phiên bản hiện tại đã hoàn thiện pipeline ingestion (đọc dữ liệu → tiền xử lý → chia đoạn → embedding → lưu ChromaDB). Các node trong pha truy vấn vẫn là placeholder cho tới khi bạn triển khai bước tiếp theo.

## 3. Chuẩn bị môi trường

### 3.1. Yêu cầu hệ thống
- Python >= 3.10
- (Khuyến nghị) GPU NVIDIA có CUDA nếu muốn tăng tốc quá trình tạo embedding hoặc suy luận mô hình lớn.
- Hệ điều hành: Linux / macOS / Windows Subsystem for Linux (WSL).

### 3.2. Tạo virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# hoặc
.venv\\Scripts\\activate        # Windows PowerShell
```

### 3.3. Cài đặt phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Các thư viện quan trọng gồm: `langgraph`, `langchain`, `langchain-google-genai`, `chromadb`, `sentence-transformers`, `streamlit`…

## 4. Cấu hình biến môi trường Gemini
1. Tạo file `.env` ở thư mục gốc của dự án (cùng cấp với `main.py`).
2. Thêm khóa API Gemini:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```
3. (Tùy chọn) Ghi đè các tham số khác bằng prefix `RAG_`. Ví dụ:
   ```env
   RAG_CHROMA_DB_ROOT="/absolute/path/to/db/chroma_instruct2ds"
   RAG_STREAMLIT_PORT=8501
   ```

> Tham khảo toàn bộ các cấu hình khả dụng trong `src/config/settings.py`.

## 5. Chuẩn bị dữ liệu Instruct2DS

1. Đảm bảo thư mục `data/` giữ nguyên cấu trúc domain như sau:
   ```
   data/
   ├── ACL/
   │   ├── paper_metadata_ACL_1979.json
   │   └── ...
   ├── CVPR/
   └── ...
   ```
2. Mỗi file JSON chứa metadata của các bài báo. Khi hiện thực node `load_documents`, bạn sẽ cần đọc các tệp này bằng `pandas` hoặc `json` để tạo danh sách `DocumentRecord`.
3. Nếu dữ liệu đặt ở vị trí khác, truyền đường dẫn mới thông qua biến môi trường `RAG_DATA_ROOT` hoặc tham số `--dataset-roots` khi chạy ingestion.

## 6. Chạy pipeline ingestion

1. **Kiểm tra môi trường ảo đã kích hoạt** (`which python` nên trỏ tới `.venv`).
2. **(Tuỳ chọn) Thiết lập embedding giả để kiểm thử nhanh:** đặt `RAG_EMBEDDING_MODEL=fake:32` để dùng mô hình nhúng giả định (không cần tải trọng số từ Hugging Face).
3. **Thực thi lệnh sau:**
   ```bash
   python main.py ingest
   ```
   - Mặc định, hệ thống sẽ sử dụng `data/` làm nguồn dữ liệu.
   - Pipeline sẽ đọc metadata, làm sạch nội dung, chia đoạn, sinh embedding và lưu vào ChromaDB.
4. **Tùy chỉnh đường dẫn dữ liệu** (khi bạn muốn giới hạn ở một vài domain cụ thể):
   ```bash
   python main.py ingest --dataset-roots data/ACL data/NeurIPS
   ```
   - `main.py` → gọi `compile_ingestion_graph` → khởi tạo `IngestionState` với danh sách thư mục.
   - Graph đã hiện thực đầy đủ chuỗi xử lý ingestion.
5. **Xác minh kết quả:** thư mục `db/chroma_instruct2ds/` (mặc định) sẽ chứa dữ liệu Chroma và log hiển thị `vector_store_status="persisted:<số chunk>"`.

## 7. Chạy pipeline truy vấn

1. Đảm bảo quá trình ingestion đã hoàn thành và vector store tồn tại.
2. **Gửi câu hỏi tiếng Việt:**
   ```bash
   python main.py query "Mô hình RAG được giới thiệu ở hội nghị nào?"
   ```
   - Hệ thống sẽ biên dịch graph truy vấn và chuẩn bị `QueryState` với câu hỏi đã nhập.
3. Khi bạn triển khai các node thật, quy trình sẽ:
   - `retrieve_context`: truy xuất top-k đoạn từ ChromaDB bằng embedding truy vấn.
   - `synthesize_answer`: gọi Gemini để sinh câu trả lời dựa trên ngữ cảnh.
   - `format_response`: chuẩn hóa câu trả lời và gắn trích dẫn.
4. Log sẽ hiển thị câu trả lời mẫu "Tính năng đang được xây dựng." cho tới khi bạn cập nhật node thực tế.

## 8. Tùy chỉnh nâng cao

- **Thay đổi mô hình embedding:** chỉnh `RAG_EMBEDDING_MODEL` trong `.env` (ví dụ `intfloat/multilingual-e5-base`).
- **Điều chỉnh số đoạn truy xuất:** cập nhật `RAG_RETRIEVER_K`.
- **Thay đổi thư mục lưu Chroma:** `RAG_CHROMA_DB_ROOT`.
- **Thêm cache mô phỏng OHCache:** hiện thực thêm node mới trong `src/nodes/query_nodes.py` và gắn vào graph bằng `graph.add_node` + `graph.add_edge`.

## 9. Kiểm thử nhanh sau khi chỉnh sửa mã

Để đảm bảo các tập tin Python biên dịch thành công:
```bash
python -m compileall src main.py
```
Lệnh này giúp phát hiện lỗi cú pháp trước khi triển khai.

## 10. Khắc phục sự cố thường gặp

| Vấn đề | Nguyên nhân khả dĩ | Cách xử lý |
| --- | --- | --- |
| `ModuleNotFoundError: langgraph` | Chưa cài đặt phụ thuộc trong môi trường ảo | Kích hoạt virtual env và chạy `pip install -r requirements.txt` |
| `ValueError: Unknown command` | Gõ sai câu lệnh `ingest`/`query` | Kiểm tra lại cú pháp `python main.py --help` |
| Không tìm thấy dữ liệu | Sai đường dẫn hoặc quyền truy cập | Cập nhật `--dataset-roots` hoặc biến `RAG_DATA_ROOT` |
| Chưa có câu trả lời thực tế | Các node truy vấn vẫn là placeholder | Hiện thực các node trong `src/nodes/query_nodes.py` |

## 11. Lộ trình phát triển tiếp theo

1. **Tối ưu pipeline ingestion**: tinh chỉnh tham số chunk, lọc trùng lặp, bổ sung cache trung gian nếu cần.
2. **Hoàn thiện node truy vấn**: truy xuất vector, gọi Gemini thông qua `langchain-google-genai`, định dạng kết quả.
3. **Tích hợp Streamlit**: tạo giao diện web ở `app.py` (hoặc tương đương) để người dùng nhập câu hỏi và xem kết quả.
4. **Cơ chế OHCache**: cài đặt bộ nhớ tạm tránh lặp embedding và truy vấn tương tự.

Khi làm theo các bước trong README này, bạn sẽ có nền tảng vững chắc để hoàn thiện hệ thống RAG end-to-end bằng LangGraph và Gemini.
