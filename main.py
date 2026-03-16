from backend.app.database import setup
from backend.app.core.ai import get_llm_and_agent

def test_agent():
    """Test agent với câu hỏi mẫu"""
    # Khởi tạo retriever và agent
    agent_executor = get_llm_and_agent()
    print("=" * 60)
    print("🤖 Agent đã sẵn sàng! Nhập 'stop' hoặc 'exit' để thoát.")
    print("=" * 60)

    while True:
        question = input("\n💬 Bạn: ").strip()

        if question.lower() in ['stop', 'exit']:
            print("\n👋 Tạm biệt! Hẹn gặp lại.")
            break

        if not question:
            print("⚠️ Vui lòng nhập câu hỏi.")
            continue

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\nAgent: {result['output']}")
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    # setup()
    test_agent()