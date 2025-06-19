import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 匯入靜態知識庫和動態日誌獲取函式
from .rag_corpus import CORPUS_DOCS
from .log_handler import get_all_logs_as_docs

class DynamicRAGSystem:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """初始化模型和索引"""
        print("Initializing Dynamic RAG System...")
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.refresh_index() # 首次初始化時建立索引

    def _build_index(self):
        """內部函式：結合靜態與動態資料來建立或重建 FAISS 索引"""
        print("Building/Refreshing RAG index...")
        static_docs = CORPUS_DOCS
        dynamic_docs = get_all_logs_as_docs()

        self.documents = static_docs + dynamic_docs

        if not self.documents:
            print("No documents found to build RAG index.")
            self.index = None
            return

        embeddings = self.embedder.encode(self.documents, normalize_embeddings=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity
        self.index.add(np.asarray(embeddings, dtype="float32"))
        print(f"RAG index built successfully with {len(self.documents)} documents.")

    def refresh_index(self):
        """公開的刷新索引方法"""
        self._build_index()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """從結合後的索引中檢索最相關的文件片段"""
        if self.index is None or not self.documents:
            return "" # 如果沒有索引，返回空字串

        query_vector = self.embedder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.asarray(query_vector, dtype="float32"), top_k)

        hits = [self.documents[i] for i in indices[0] if i != -1]
        return "\n\n".join(hits)

# 建立一個全域實例，方便在應用程式中重用
rag_system = DynamicRAGSystem()