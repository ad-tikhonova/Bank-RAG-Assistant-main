import os
import logging
import shutil
from typing import List

# Стабильные импорты
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- 1. ЛОГИРОВАНИЕ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. КОНФИГУРАЦИЯ ---
DATA_PATH = "knowledge_base"
DB_PATH = "./neostack_db"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class NeoStackRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore = None
        self.bm25_retriever = None

    def prepare_database(self):
        logger.info("Загрузка документов...")
        loader = DirectoryLoader(DATA_PATH, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(doc.metadata.get("source", "unknown"))

        chunks = self.text_splitter.split_documents(documents)
        if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
        
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=DB_PATH)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 2
        return chunks

    def hybrid_search(self, query: str) -> List[Document]:
        """Гибридный поиск (Этап 2)"""
        # Векторный поиск
        vec_results = self.vectorstore.similarity_search(query, k=2)
        # Поиск по ключевым словам (BM25)
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Слияние
        combined = vec_results + bm25_results
        unique_results = []
        seen_content = set()
        for doc in combined:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
        return unique_results[:3]

# --- 3. ЭТАП 4: МЕТРИКИ ---
def evaluate_retrieval(rag_system):
    print("\n" + "="*50)
    print("ЭТАП 4: ОЦЕНКА КАЧЕСТВА ПОИСКА (METRICS)")
    print("="*50)
    
    ground_truth = [
        {"query": "справка 2-НДФЛ", "expected": "credit_universal_terms.txt"},
        {"query": "потерял карту", "expected": "bank_fq_support.txt"},
        {"query": "ставка для молодежи", "expected": "junior-investor.txt"},
        {"query": "реферальная программа", "expected": "loyalty.txt"}
    ]

    hits = 0
    mrr_score = 0
    for item in ground_truth:
        results = rag_system.hybrid_search(item["query"])
        sources = [doc.metadata.get("source_file") for doc in results]
        
        if item["expected"] in sources:
            hits += 1
            rank = sources.index(item["expected"]) + 1
            mrr_score += 1 / rank
        print(f"Запрос: {item['query']} | Найдено: {sources}")

    print(f"\nИТОГ: Hit Rate@3 = {(hits/len(ground_truth))*100}% | MRR = {mrr_score/len(ground_truth):.2f}")

# --- 4. ЗАПУСК ---
if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        rag = NeoStackRAG()
        rag.prepare_database()
        
        # Тестовый запрос (Этап 3)
        print("\n--- ТЕСТОВЫЙ ЗАПРОС ---")
        res = rag.hybrid_search("Нужна ли справка 2-НДФЛ?")
        print(f"Ответ найден в: {res[0].metadata['source_file']}")
        
        # Метрики (Этап 4)
        evaluate_retrieval(rag)