from datetime import datetime
import os
from langchain_core.tools import tool
from backend.app.core.ai import get_retriever
from firecrawl import FirecrawlApp
from backend.utils.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



tavily = TavilySearchResults(
    api_key=Settings.tavily_api_key,
    max_results=3,
    include_domains=["huce.edu.vn"]
)

firecrawl = FirecrawlApp(api_key=Settings.firecrawl_api_key)
embedding_model = HuggingFaceEmbeddings(model_name=Settings.EMBEDDING_MODEL)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)
reranker_model = HuggingFaceCrossEncoder(model_name=Settings.RERANK_MODEL)
reranker = CrossEncoderReranker(model=reranker_model, top_n=3)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, reranker])

# DOC_MAPPING = {
#     "hoạt động chung": "https://huce.edu.vn/hoat-dong-chung",
#     "tin đào tạo": "https://huce.edu.vn/tin-tuc-dao-tao",
#     "tin sinh viên": "https://huce.edu.vn/tin-tuc-sinh-vien",
#     "nghiên cứu khoa học": "https://huce.edu.vn/tin-tuc-nghien-cuu-khoa-hoc",
#     "hợp tác quốc tế": "https://huce.edu.vn/tin-tuc-hop-tac-quoc-te",
#     "báo chí": "https://huce.edu.vn/xay-duntg-ha-noi-tren-bao-chi",
#     "đào tạo sinh viên": "https://sinhvien.huce.edu.vn/sinh-vien/dm-tin-tuc/dao-tao.html",
#     "quy chế": "https://sinhvien.huce.edu.vn/sinh-vien/dm-tin-tuc/quy-che-quy-dinh.html",
#     "sự kiện": "https://huce.edu.vn/su-kien",
#     "thông báo": "https://huce.edu.vn/thong-bao"
# }


@tool
def search_huce(query: str) -> str:
    """Tìm kiếm nhanh: 'thông báo học bổng tháng 3'

    Tavily search Google -> Trả về danh sách links + snippet.
    """
    return tavily.invoke({"query": f"{query} site:huce.edu.vn"})

@tool
def extract_huce_page(url: str) -> str:
    """Đọc toàn bộ nội dung 1 trang: 'https://huce.edu.vn/quy-che'

    Firecrawl vào trang -> Lấy full text markdown sạch.
    """
    try:
        result = firecrawl.scrape(url,
                                  formats = ['markdown', 'html'],
                                  remove_base64_images=True,
                                  block_ads=True,
                                  only_main_content=True)
        return result.markdown
    except Exception as e:
        return f"Lỗi: {str(e)}"


@tool
def knowledge_retrieval_tool(query: str, collection_name: str = "data_test") -> str:
    """
    Công cụ tìm kiếm thông tin từ cơ sở tri thức của trường.

    Args:
        query: Câu hỏi cần tìm
        collection_name: Tên bộ sưu tập (mặc định: data_test)
    """
    try:
        # 1. Lấy Base Retriever (Retriever gốc)
        base_retriever = get_retriever(collection_name)

        # 2. Bọc lớp nén/lọc (Wrapper)
        # ContextualCompressionRetriever sẽ lấy docs từ base, sau đó chạy qua filter để loại bỏ cái trùng
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever
        )

        # 3. Gọi invoke trên retriever đã bọc
        docs = compression_retriever.invoke(query)

        if not docs:
            return "[Suy luận] Không tìm thấy thông tin chắc chắn. Bạn có thể hỏi cụ thể hơn không?"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            # Thêm logic làm sạch xuống dòng thừa nếu cần
            content = doc.page_content.strip().replace('\n\n', '\n')
            results.append(f"[Nguồn {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"[Lỗi] Không thể truy vấn: {str(e)}"


@tool
def get_current_time() -> str:
    """Lấy thời gian hiện tại.

    Returns:
        str: Thời gian hiện tại theo định dạng Việt Nam
    """
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


@tool
def get_current_date() -> str:
    """Lấy ngày hiện tại.

    Returns:
        str: Ngày hiện tại theo định dạng DD/MM/YYYY
    """
    return datetime.now().strftime("%d/%m/%Y")


@tool
def get_current_weekday() -> str:
    """Lấy thứ trong tuần hiện tại.

    Returns:
        str: Thứ trong tuần bằng tiếng Việt
    """
    weekdays = {
        0: "Thứ Hai",
        1: "Thứ Ba",
        2: "Thứ Tư",
        3: "Thứ Năm",
        4: "Thứ Sáu",
        5: "Thứ Bảy",
        6: "Chủ Nhật"
    }
    return weekdays[datetime.now().weekday()]

ALL_TOOLS = [knowledge_retrieval_tool,
             get_current_time,
             get_current_date,
             get_current_weekday,
             extract_huce_page,
             search_huce
             ]