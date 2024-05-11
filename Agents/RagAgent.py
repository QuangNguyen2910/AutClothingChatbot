from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from typing import Optional, List, Tuple
from transformers import AutoTokenizer

def document_chunking(files: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        length_function = len,
        is_separator_regex = False,
    )

    docs_processed = []
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            docs_processed += text_splitter.split_documents([LangchainDocument(page_content = text)])
    
    return docs_processed

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = "thenlper/gte-small",
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

