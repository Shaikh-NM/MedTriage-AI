import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

INDEX_PATH = "app/rag/faiss_index"


class MedicalRetriever:
    def __init__(self, index_path: str = INDEX_PATH):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def retrieve_with_metadata(self, query: str, k: int = 5) -> list[dict]:
        """Return content alongside source metadata for richer agent context."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "title": doc.metadata.get("title", ""),
                "section": doc.metadata.get("section", ""),
                "url": doc.metadata.get("url", ""),
            }
            for doc in docs
        ]
