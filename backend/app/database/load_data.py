import os
from llama_parse.utils import ResultType
from langchain_community.document_loaders import TextLoader
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from dotenv import load_dotenv
load_dotenv()



def load_data_from_folder(folder_path="./backend/data"):
    """
    Quét toàn bộ thư mục và đọc tất cả các file file .txt, .docx, .pdf
    sau đó chia nhỏ nội dung thành các Document bằng RecursiveCharacterTextSplitter.
    """
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    llama_parser = LlamaParse(result_type=ResultType.MD,
                              language="vi",
                              verbose=True)

    all_chunks = []
    cloud_extensions = (".pdf", ".docx", ".png", ".jpg", ".jpeg")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        doc_name = filename.rsplit('.', 1)[0].replace('_', ' ')

        if filename.lower().endswith(cloud_extensions):
            print(f"🦇 Đang bóc tách {filename} qua Cloud LlamaParse...")
            parsed_docs = llama_parser.load_data(file_path)

            full_md_text = "\n".join([doc.text for doc in parsed_docs])
            md_chunks = md_splitter.split_text(full_md_text)

            for chunk in md_chunks:
                chunk.metadata['doc_name'] = doc_name
                chunk.metadata['source'] = filename

            all_chunks.extend(md_chunks)


        elif filename.endswith(".txt"):
            print(f"🦇 Đang bóc tách {filename} bằng Local Loader...")
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            txt_chunks = txt_splitter.split_documents(docs)

            # Bổ sung metadata
            for chunk in txt_chunks:
                chunk.metadata['doc_name'] = doc_name
                chunk.metadata['source'] = filename

            all_chunks.extend(txt_chunks)

    return all_chunks