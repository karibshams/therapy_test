import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import PyPDF2
from pdfplumber import PDF

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PDFDocument:
    filename: str
    content: str
    metadata: Dict
    page_count: int

class PDFVectorStore:
    def __init__(self, folder_path: str = "./pdf/"):
        self.folder_path = folder_path
        self.documents: List[PDFDocument] = []
        self.vector_store: Optional[FAISS] = None
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        os.makedirs(folder_path, exist_ok=True)
        
    def load_pdf_files(self) -> List[PDFDocument]:
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.folder_path}")
            return []
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.folder_path, pdf_file)
            try:
                content = self._extract_with_pdfplumber(file_path)
                if not content:
                    content = self._extract_with_pypdf2(file_path)
                if content:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        page_count = len(pdf_reader.pages)
                    doc = PDFDocument(
                        filename=pdf_file,
                        content=content,
                        metadata={'source': pdf_file, 'therapy_type': 'general'},
                        page_count=page_count
                    )
                    self.documents.append(doc)
                    logger.info(f"Successfully loaded: {pdf_file} ({page_count} pages)")
                else:
                    logger.error(f"Could not extract content from: {pdf_file}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
        return self.documents

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        try:
            with PDF.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return ""

    def _extract_with_pypdf2(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""

    def build_vector_store(self) -> FAISS:
        try:
            if not self.documents:
                self.load_pdf_files()
            if not self.documents:
                raise ValueError("No documents loaded. Please add PDF files to the folder.")
            
            langchain_docs = []
            for pdf_doc in self.documents:
                chunks = self.text_splitter.split_text(pdf_doc.content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={**pdf_doc.metadata, 'chunk_id': i, 'total_chunks': len(chunks)}
                    )
                    langchain_docs.append(doc)
            
            self.vector_store = FAISS.from_documents(documents=langchain_docs, embedding=self.embeddings)
            self.save_vector_store()
            logger.info("Vector store successfully built and saved.")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise

    def save_vector_store(self, path: str = "./vector_store/"):
        try:
            if self.vector_store:
                os.makedirs(path, exist_ok=True)
                self.vector_store.save_local(path)
                logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load_vector_store(self, path: str = "./vector_store/", allow_dangerous_deserialization: bool = False):
        if os.path.exists(path):
            try:
                self.vector_store = FAISS.load_local(
                    path,
                    self.embeddings,
                    allow_dangerous_deserialization=allow_dangerous_deserialization
                )
                logger.info(f"Vector store loaded from {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return False
        return False

    def get_stats(self):
        return {
            "total_pdfs": len(self.documents),
            "total_chunks": sum(len(self.text_splitter.split_text(doc.content)) for doc in self.documents)
        }
    
    def retrieve_pdf_context(self, query: str, top_k: int = 3) -> str:
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search(query, k=top_k)
        combined_text = "\n---\n".join([doc.page_content for doc in results])
        return combined_text
