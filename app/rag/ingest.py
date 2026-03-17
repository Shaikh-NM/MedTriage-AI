import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "app/rag/data/who_guidelines"
INDEX_PATH = "app/rag/faiss_index"

# Sections that contain only navigation/boilerplate noise — skip them
NOISE_HEADINGS = {"", "Introduction", "Database", "Related health topic"}


def load_who_documents(data_dir: str) -> list[Document]:
    """
    Parse every WHO fact-sheet JSON and return one Document per content section.
    The "Introduction" section in every file is the site navigation menu and is
    excluded. Short boilerplate headings are also skipped.
    """
    docs: list[Document] = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        title: str = data.get("title", "")
        url: str = data.get("url", "")

        for section in data.get("sections", []):
            heading: str = section.get("heading", "").strip()
            content: str = section.get("content", "").strip()

            if not content or heading in NOISE_HEADINGS:
                continue

            # Format: title + heading + body so every chunk is self-contained
            page_content = f"{title}\n\n## {heading}\n\n{content}"

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "title": title,
                        "section": heading,
                        "url": url,
                        "source": filename,
                    },
                )
            )

    return docs


def ingest_documents(
    data_dir: str = DATA_DIR,
    index_path: str = INDEX_PATH,
) -> None:
    print(f"Loading WHO fact-sheet documents from: {data_dir}")
    docs = load_who_documents(data_dir)
    print(f"  Loaded {len(docs)} sections across all fact sheets")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    print("Generating embeddings with OpenAI (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"  FAISS index saved to: {index_path}")
    print(f"Done. {len(chunks)} chunks ingested.")


if __name__ == "__main__":
    ingest_documents()
