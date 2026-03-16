# Import các thư viện cần thiết
from backend.app.database.seed_data import connect_to_chroma
from langchain_classic.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document
from dotenv import load_dotenv
load_dotenv()


def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever | BM25Retriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Chroma) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        # Kết nối với Milvus và tạo vector retriever
        vectorstore = connect_to_chroma(collection_name)
        chroma_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        # Tạo BM25 retriever từ toàn bộ documents
        import pickle
        with open(f"D:/Project/huce-assistant/backend/app/database/cache/{collection_name}_docs.pkl", "rb") as f:
            documents = pickle.load(f)

        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5

        # Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.8, 0.2]
        )
        return ensemble_retriever

    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Fallback trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)
