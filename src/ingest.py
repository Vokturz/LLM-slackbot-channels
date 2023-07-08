# This has been copied from https://github.com/imartinez/privateGPT/blob/main/ingest.py
import os
import glob
from typing import List, Optional
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

def load_single_document(file_path: str, pretty_type: str) -> Optional[Document]:
    """
    Load a single document from a file
    
    Args:
        file_path: The path to the file to load.
        pretty_type: The type of document to load.

    Returns:
        doc: The loaded document.
    """
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
                      chunk_size: int, chunk_overlap: int,
                      extra_separators: List[str]) -> List[Document]:
    """
    Load documents and split them into chunks.

    Args:
        documents: The documents to load.
        chunk_size: The chunk size fo the text splitter.
        chunk_overlap: The chunk overlap for the text splitter.
        extra_separators: The extra separators to use.

    Returns:
        texts: The documents split into chunks.
    """
    separators = extra_separators + ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   separators=separators
                                                   )
    texts = text_splitter.split_documents(documents)
    return texts