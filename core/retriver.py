import os
import pandas as pd
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from langchain.schema import Document
from core.logging import LoggerFactory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Initialize logger
logger = LoggerFactory().get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")


class Retriever:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_match: int = 1,
        relevence_threshold: float = 0.7,
        relevent_candidates: int = 10,
        embedding_model: Optional[str] = None,
    ):
        self.df = df
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_match = top_match
        self.relevence_threshold = relevence_threshold
        self.relevent_candidates = relevent_candidates
        self.model_name = embedding_model or os.getenv("EMBEDDING_MODEL")
        self.embedding = None
        self.splitter = None
        self.corpus: List[Document] = []
        self.chunks: List[Document] = []

    def _init_embedding(self):
        self.embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        logger.info(f"Embedding initialized: {self.model_name}")

    def _init_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        logger.info(
            f"Splitter initialized | chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def _create_corpus(self) -> List[Document]:
        if self.df is None:
            raise ValueError("DataFrame is required to create corpus")

        documents = [
            Document(
                page_content=row["article"],
                metadata={
                    "article_publish_date": row["date"],
                    "article_title": row["title"],
                    "article_link": row["link"],
                },
            )
            for _, row in self.df.iterrows()
        ]

        logger.info(f"Corpus created: {len(documents)} documents")
        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Documents chunked: {len(chunks)} chunks")
        return chunks

    def create_retirver(self) -> VectorStoreRetriever:
        logger.info("Building retriever pipeline...")

        self._init_embedding()
        self._init_splitter()

        self.corpus = self._create_corpus()
        self.chunks = self._chunk_documents(self.corpus)

        self.retriver = FAISS.from_documents(self.chunks, self.embedding).as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.top_match,
                "fetch_k": self.relevent_candidates,
                "lambda_mult": self.relevence_threshold,
            },
        )
        logger.info("Retriever build complete")
        return self.retriver
