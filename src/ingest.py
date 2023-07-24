# The first two functions has been copied from
# https://github.com/imartinez/privateGPT/blob/main/ingest.py

from typing import List, Optional
from dotenv import load_dotenv
from typing import (Dict, Any)
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
from pathlib import Path
import os
import glob

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

def process_uploaded_files(files: Dict[str, Any], 
                           bot_token: str,
                           chunk_size: int, chunk_overlap: int,
                           extra_separators: List[str], 
                            ) -> List[Document]:
    """
    Process the files uploaded by the user. All files are splitted using the
    functions from the ingest module and the bot configuration. This is then
    used to create a QA thread.


    Args:
        bot_token: The bot token.
        files: The files uploaded by the user as a Slack response.
        chunk_size: The chunk size for the text splitter.
        chunk_overlap: The chunk overlap for the text splitter.
        extra_separators: The extra separators to use.

    Returns:
        texts: The documents split into chunks.
    """
    all_texts = []
    for _file in files:
        url = _file['url_private_download']
        file_name = _file['name'] 
        pretty_type = _file['pretty_type']
        doc_list = []
        if '.'+file_name.split('.')[-1] in LOADER_MAPPING.keys():
            resp = requests.get(url, headers={'Authorization':
                                                'Bearer %s' % bot_token})
            save_file = Path(f'data/tmp/{file_name}')
            save_file.write_bytes(resp.content)
            doc_list.append(load_single_document(f'data/tmp/{file_name}',
                                                        pretty_type))
            save_file.unlink()
        texts = process_documents(doc_list, chunk_size=chunk_size,
                                  chunk_overlap=chunk_overlap,
                                  extra_separators=extra_separators)  
        all_texts.extend(texts) 
    return all_texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(persist_directory) and os.path.exists(os.path.join(persist_directory, 'chroma.sqlite3')):
        list_index_files = glob.glob(os.path.join(persist_directory, '*/*'))
         # At least 4 files are needed in a working vectorstore
        if len(list_index_files) > 3:
            return True
    return False