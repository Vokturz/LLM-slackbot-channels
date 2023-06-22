# This has been copied from https://github.com/imartinez/privateGPT/blob/main/ingest.py
import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
#from constants import CHROMA_SETTINGS


load_dotenv()


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str, pretty_type: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        doc = loader.load()[0]
        doc.metadata["pretty_type"] = pretty_type
        return doc
    else:
        pass

def process_documents(documents: List[Document],
                      chunk_size: int, chunk_overlap:
                      int) -> List[Document]:
    """
    Load documents and split in chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts