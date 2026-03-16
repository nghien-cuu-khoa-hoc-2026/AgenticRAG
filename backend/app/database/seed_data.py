import os
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from backend.app.database.load_data import load_data_from_folder
import chromadb
from huggingface_hub import login as hf_login
from langchain_chroma import Chroma
from uuid import uuid4
from backend.utils.config import Settings
from dotenv import load_dotenv
load_dotenv()

hf_login(token=os.getenv('HF_TOKEN'))

def seed_chroma(collection_name: str, directory: str) -> Chroma:
    """
    Hàm tạo và lưu vector embeddings vào Chroma từ dữ liệu local
    Args:
        collection_name (str): Tên collection trong Milvus để lưu dữ liệu
        directory (str): Thư mục chứa file dữ liệu
    """
    # Khởi tạo model embeddings tùy theo lựa chọn
    embeddings = HuggingFaceEmbeddings(model_name=Settings.EMBEDDING_MODEL)

    # Đọc dữ liệu từ file local
    documents = load_data_from_folder(directory)

    # Tạo ID duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Khởi tạo và cấu hình Chroma
    vectorstore = Chroma(
        collection_name=collection_name,
        client=chromadb.HttpClient(host="localhost", port=8080),
        embedding_function=embeddings,
    )
    # Thêm documents vào Chroma
    vectorstore.from_documents(documents=documents, ids=uuids)

    print(f"Đã seed {len(documents)} documents vào collection '{collection_name}'")
    cache_path = f"D:/Project/huce-assistant/backend/app/database/cache/{collection_name}_docs.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)


    return vectorstore

def setup():
    """Seed data lần đầu (chỉ chạy 1 lần)"""
    print("🔧 Đang seed dữ liệu vào ChromaDB...")
    seed_chroma(
        collection_name="data_test",
        directory="./backend/data"  # Bỏ file PDF/DOCX vào đây
    )
    print(" Setup hoàn tất!\n")


def connect_to_chroma(collection_name: str) -> Chroma:
    """
    Hàm kết nối đến collection có sẵn trong Chroma
    Args:
        collection_name (str): Tên collection cần kết nối
    Returns:
        Chroma: Đối tượng Chroma đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng EMBEDDING MODEL cho việc tạo embeddings khi truy vấn
    """
    embeddings = HuggingFaceEmbeddings(model_name=Settings.EMBEDDING_MODEL)

    try:
        client = chromadb.HttpClient(host="localhost", port=8080)
        client.heartbeat()  # Test connection

        vectorstore = Chroma(
            embedding_function=embeddings,
            client=client,
            collection_name=collection_name,
        )
        return vectorstore

    except Exception as e:
        print(f"Không thể kết nối ChromaDB: {e}")
        print("Hãy chạy: docker run -p 8080:8000 chromadb/chroma")
        raise