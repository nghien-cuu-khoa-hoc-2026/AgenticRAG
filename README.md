## ChromaDB Server: Khởi động, tái sử dụng dữ liệu, và khi nào cần seed lại

### BƯỚC 1: Khởi động ChromaDB Server (có volume để lưu data)

```bash
docker run -d \
  --name chromadb_server \
  -p 8080:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma
```

### Kiểm tra server đã chạy chưa

```bash
curl http://localhost:8080/api/v2/heartbeat
```

Kết quả mẫu:

```json
{"nanosecond heartbeat": 1738483200000000000}
```

---

### BƯỚC 1: Khởi động lại container (nếu đã stop)

```bash
docker start chromadb_server
```

Hoặc nếu đã xóa container, chạy lại với **CÙNG volume**:

```bash
docker run -d \
  --name chromadb_server \
  -p 8080:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma
```

### BƯỚC 2: Connect và dùng ngay (**KHÔNG CẦN seed lại**)

```bash
python main.py
```

> Lưu ý: bỏ `#setup()`.

---

## 🎯 KHI NÀO CẦN SEED LẠI?

### 1) Có dữ liệu/tài liệu mới

Ví dụ: trường ra quy chế mới  
→ Bỏ file PDF mới vào `./backend/data/` rồi chạy lại seed:

```python
seed_chroma("student_handbook", "./backend/data")
```

### 2) Thay đổi cách chunk văn bản

Ví dụ: đổi `chunk_size` từ `1000` → `500`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # ✏️ Thay đổi
    chunk_overlap=100
)
```

→ **Phải seed lại** để áp dụng cách chia mới.

### 3) Đổi embedding model

Ví dụ: chuyển từ AITeamVN sang model khác:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"  # Model mới
)
```

→ **Phải seed lại** vì vector dimensions khác.

### 4) Xóa collection và làm lại từ đầu

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8080)
client.delete_collection("data_test")  # Xóa
```
