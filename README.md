# BƯỚC 1: Khởi động ChromaDB Server (có volume để lưu data)
docker run -d \
  --name chromadb_server \
  -p 8080:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma

# Kiểm tra server đã chạy chưa
curl http://localhost:8080/api/v2/heartbeat
# Kết quả: {"nanosecond heartbeat": 1738483200000000000}



# BƯỚC 1: Khởi động lại container (nếu đã stop)
docker start chromadb_server

# Hoặc nếu đã xóa container, chạy lại với CÙNG volume
docker run -d \
  --name chromadb_server \
  -p 8080:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma

# BƯỚC 2: Connect và dùng ngay (KHÔNG CẦN SEED LẠI)
python backend/app/retriever_agent.py



# 🎯 KHI NÀO CẦN SEED LẠI?

## 1.
# Ví dụ: Trường ra quy chế mới
# Bỏ file PDF mới vào ./backend/data/
# Chạy lại seed
seed_chroma("student_handbook", "./backend/data")

## 2.
# Thay đổi chunk_size từ 1000 → 500
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # ✏️ Thay đổi
    chunk_overlap=100
)
# Phải seed lại để áp dụng cách chia mới

## 3.
# Chuyển từ AITeamVN sang model khác
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"  #  Model mới
)
# Phải seed lại vì vector dimensions khác



## 4. Xóa collection và làm lại từ đầu
import chromadb

client = chromadb.HttpClient(host="localhost", port=8080)
client.delete_collection("data_test")  # Xóa

# Sau đó seed lại
seed_chroma("student_handbook", "./backend/data")
