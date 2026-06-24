import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


RAW_DATA_DIR = Path("/home/lyj/Agent/react_agent+rag/data/raw")
VECTOR_DB_DIR = Path("/home/lyj/Agent/react_agent+rag/data/vector_db/mixed_docs_faiss")

EMBEDDING_MODEL_PATH = "/home/lyj/sentence-transformers/BAAI/bge-small-zh-v1.5"


def load_txt_file(file_path: Path):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
        doc.metadata["file_type"] = "txt"

    return docs

def load_md_file(file_path: Path):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
        doc.metadata["file_type"] = "md"

    return docs

def load_pdf_file(file_path: Path):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
        doc.metadata["file_type"] = "pdf"

    return docs


def load_all_documents():
    all_docs = []

    for file_path in RAW_DATA_DIR.rglob("*"):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            print(f"加载 TXT 文件：{file_path}")
            docs = load_txt_file(file_path)
            all_docs.extend(docs)

        elif suffix == ".pdf":
            print(f"加载 PDF 文件：{file_path}")
            docs = load_pdf_file(file_path)
            all_docs.extend(docs)
        elif suffix == ".md":
            print(f"加载 Markdown 文件：{file_path}")
            docs = load_md_file(file_path)
            all_docs.extend(docs)

        else:
            print(f"跳过不支持的文件：{file_path}")

    return all_docs


def build_vector_db():
    docs = load_all_documents()

    if not docs:
        print("没有加载到任何文档")
        return

    print(f"原始文档数量：{len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )

    split_docs = text_splitter.split_documents(docs)

    print(f"切分后文档块数量：{len(split_docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(
        split_docs,
        embeddings
    )

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    db.save_local(str(VECTOR_DB_DIR))

    # 保存切分后的文档，供 BM25 混合检索使用
    chunks_path = VECTOR_DB_DIR / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(split_docs, f)
    print(f"文档块已保存到：{chunks_path}（{len(split_docs)} 块）")

    print(f"向量库已保存到：{VECTOR_DB_DIR}")


if __name__ == "__main__":
    build_vector_db()