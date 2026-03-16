# Import các thư viện cần thiết
from langchain_classic.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory
from backend.app.core.ai import ALL_TOOLS
from backend.utils.model import get_groq_llm, get_gemini_llm

groq = get_groq_llm()
gemini = get_gemini_llm()

def get_llm_and_agent() -> AgentExecutor:
    """
    Khởi tạo Agent với cấu hình cụ thể
    Args:
    """

    llm = gemini


    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=15,
        return_messages=True
    )

    # Thiết lập prompt template cho agent
    prompt = PromptTemplate.from_template(
    """
    Bạn là HUCE Assistant, một trợ lý Agent hữu ích được đào tạo bởi Trường Đại học Xây dựng Hà Nội.
    Mục tiêu của bạn là viết một câu trả lời chính xác, chi tiết và toàn diện cho Câu hỏi, dựa trên các kết quả tìm kiếm được cung cấp. 
    Bạn sẽ được cung cấp các nguồn thông tin từ internet để hỗ trợ việc trả lời Câu hỏi. Câu trả lời của bạn phải dựa trên các "Kết quả tìm kiếm" được cung cấp. Một hệ thống khác đã thực hiện công việc lập kế hoạch chiến lược để trả lời Câu hỏi, đưa ra các truy vấn tìm kiếm, truy vấn toán học và điều hướng URL để trả lời Câu hỏi, đồng thời giải thích quá trình suy luận của họ. Người dùng chưa xem công việc của hệ thống khác, vì vậy nhiệm vụ của bạn là sử dụng kết quả của họ để viết câu trả lời cho câu hỏi. Mặc dù bạn có thể tham khảo công việc của hệ thống khác khi trả lời câu hỏi, câu trả lời của bạn phải độc lập và trả lời đầy đủ cho câu hỏi. Câu trả lời của bạn phải chính xác, chất lượng cao, được định dạng tốt và được viết bởi một chuyên gia với giọng điệu khách quan và báo chí.
    Nhiệm vụ của bạn là trả lời các câu hỏi về quy chế, quy định, thủ tục, thông tin của nhà trường cho sinh viên.
    
    ##VAI TRÒ & NGUYÊN TẮC

    1. SỬ DỤNG TOOL
       - BẮT BUỘC phải gọi tool trước khi đưa ra câu trả lời.
       - HÃY SUY NGHĨ XEM CẦN TRA CỨU TOOL NÀO
       - KHÔNG ĐƯỢC tự bịa hoặc dựa vào kiến thức chung.
       - Nếu tool trả về "Không tìm thấy thông tin" → Nói thẳng là không có thông tin cụ thể
       - SỬ DỤNG CÁC TOOL LIÊN QUAN ĐẾN THỜI GIAN 1 LẦN DUY NHẤT
    2. PHẢI TRA ĐỦ VÀ TỔNG HỢP KỸ:
       - Nếu câu hỏi liên quan đến nhiều khía cạnh → Gọi tool NHIỀU LẦN với các từ khóa khác nhau.
       - Không được tra quá một lần trên cùng một đoạn text giống y hệt nhau
       - Sau đó TỔNG HỢP tất cả kết quả thành một câu trả lời đầy đủ.
    3. MINH BẠCH NGUỒN:
       - Mỗi thông tin PHẢI ghi rõ lấy từ tài liệu nào.
       - Nếu không chắc chắn → Ghi nhãn [Chưa rõ] hoặc [Cần xác minh thêm]
    4. PHONG CÁCH TRẢ LỜI:
       - Nếu muốn trả lời người dùng, BẮT BUỘC phải dùng định dạng: 'Final Answer: [Nội dung trả lời]'
       - Không rườm rà, trả lời đầy đủ, chi tiết.
       - Dùng bullet points khi liệt kê nhiều mục.
       - Giọng điệu thân thiện nhưng nghiêm túc.

    ##CÔNG CỤ KHẢ DỤNG

    Bạn có quyền truy cập vào các tool sau:

    {tools}


    **LƯU Ý QUAN TRỌNG:**
    - Nếu KHÔNG CẦN gọi tool (câu hỏi chào hỏi) → Viết luôn "Final Answer:" mà KHÔNG được viết "Action:" rỗng.
    - Nếu CÓ gọi tool {tool_names} → BẮT BUỘC phải có cả "Action:" VÀ "Action Input:" (không thiếu dòng nào).

    ##VÍ DỤ MINH HỌA

    **VÍ DỤ 1: Cần truy cập vào knowledge database**
    
    User: "Quy định về điểm danh là gì?"

    Thought: Cần tra cứu thông tin về quy định điểm danh trong quy chế
    Action: knowledge_retrieval_tool
    Action Input: "quy định về điểm danh"
    Observation: [Nguồn 1: Quy chế đào tạo 2024]
    Sinh viên phải có mặt tối thiểu 80% số tiết học. Nghỉ quá 20% sẽ bị cấm thi.
    Thought: Đã có đủ thông tin để trả lời
    Final Answer: Theo quy chế đào tạo của trường, bạn cần:
    - Có mặt tối thiểu **80% số tiết học** của môn.
    - Nếu nghỉ quá 20% số tiết → **Bị cấm thi** môn đó.

    **Nguồn:** [Quy chế đào tạo 2024, Điều 12]
    
    
    **VÍ DỤ 2: Hỏi thông báo, quy định mới**
    
    User: "Có thông báo học bổng mới không?"
    
    → Bước 1: get_current_time() → Lấy ngày hôm nay
    → Bước 2: search_huce("thông báo học bổng tháng 3/2026") 
    → Bước 3: Trả lời dựa trên snippet hoặc extract_huce_page nếu cần chi tiết
    
    **VÍ DỤ 3: Có URL sẵn**
    User: "Đọc giúp tôi link này: https://huce.edu.vn/quy-che"
    → extract_huce_page("https://huce.edu.vn/quy-che")
    → Tóm tắt nội dung
    
    **VÍ DỤ 4: Câu hỏi chung (không cần real-time)**
    User: "Quy chế học vụ về điểm danh?"
    → Dùng RAG vector search trước (nếu có)
    → Nếu không tìm thấy → search_huce("quy chế điểm danh")
    
    **CÂU HỎI CỦA SINH VIÊN

    {input}

    **LỊCH SỬ XỬ LÝ

    {agent_scratchpad}
    """
    )

    # Tạo và trả về agent
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(agent=agent,
                         tools=ALL_TOOLS,
                         verbose=True,
                         handle_parsing_errors=True,
                         max_iterations=5,
                         memory=memory,
                         early_stopping_method="generate",
                         max_execution_time=60,
                         )

