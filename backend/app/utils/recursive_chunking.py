# # import re 
# # from typing import List


# # try :
# #     import tiktoken #for coverting text to tokens and counting them, very efficient and accurate for LLMs
# #     enc=tiktoken.get_encoding("cl100k_base") # this encoding is used by OpenAI's models, so it should work well for counting tokens in a way that matches how the model will see them
# #     def count_tokens(text:str) -> int:
# #         return len(enc.encode(text))# enc.encode(text) will convert the text into a list of token IDs, and len() will give us the number of tokens. This is more accurate than just splitting on whitespace, especially for languages with complex tokenization rules or for texts with lots of punctuation.
# # except :
# #     def count_tokens(text:str) -> int:
# #         # return len(re.findall(r'\w+', text)) # can use this also
# #         return len(text.split())
    

# # try :
# #     import nltk
# #     # nltk.download('punkt')
# #     from nltk.tokenize import sent_tokenize
# #     def split_into_sentences(text:str) -> List[str]:
# #         return sent_tokenize(text)
# # except :
# #     # def split_into_sentences(text:str) -> List[str]:
# #     #     # A simple regex-based sentence splitter
# #     #     sentence_endings = re.compile(r'[.!?]+')
# #     #     sentences = sentence_endings.split(text)
# #     #     return [s.strip() for s in sentences if s.strip()]
# #     def split_sentences(text: str) -> List[str]:
# #         return re.split(r'(?<=[.!?])\s+', text)#splitting sentences based on punctuation followed by whitespace, this is a common way to split sentences in English. It looks for periods, exclamation points, or question marks followed by one or more whitespace characters, and splits the text at those points.



# # def split_headings(text:str) -> List[str]:
   
#     pattern = r'(?=\n#{1,6}\s)|(?=\n[A-Z][A-Z\s]{5,}\n)'#using regex to find pattern which can form headings
#     sections = re.split(pattern, text)
#     return [s.strip() for s in sections if s.strip()]#saving in list and removing empty sections and extra whitespace


# def split_paragraphs(text:str) -> List[str]:
#     # Split text into paragraphs based on double newlines
#     paragraphs = text.split('\n\n')
#     return [p.strip() for p in paragraphs if p.strip()]


# # def build_chunks(
# #         text:str,
# #         max_tokens:int = 500,
# #         overlap_ratio:float = 0.12
# # )-> List[str]:
# #     chunks=[]
# #     overlap_tokens=int(max_tokens * overlap_ratio)
# #     sections=split_headings(text)
# #     for section in sections:#selecting a section based on heading 
# #         paragraphs = split_paragraphs(section)#spliting section into paras

# #         current_chunk = ""
# #         current_tokens = 0

# #         for para in paragraphs:
# #             para_tokens = count_tokens(para)#counting tokens in para to check if it can fit in current chunk or not

# #             # If paragraph itself too big → split into sentences
# #             if para_tokens > max_tokens:#converting a big chunk into smaller that can fir into the maxtoken 
# #                 sentences = split_sentences(para)#again splitting para into sentences

# #                 for sent in sentences:
# #                     sent_tokens = count_tokens(sent)#counting tokens in sentence to check if it can fit in current chunk or not

# #                     if current_tokens + sent_tokens <= max_tokens:#again checking if sentence can fit in current chunk or not, if it can fit then add it to current chunk and update token count
# #                         current_chunk += " " + sent
# #                         current_tokens += sent_tokens
# #                     else:
# #                         chunks.append(current_chunk.strip())#removing extra whitespace and saving the current chunk to the list of chunks

# #                         # overlap handling
# #                         overlap_text = get_overlap_text(current_chunk, overlap_tokens)#getoverlaodtext split the text in current chunk to small chunk of overlaping token  limit and combine them into a string 
# #                         current_chunk = overlap_text + " " + sent
# #                         current_tokens = count_tokens(current_chunk)

# #             else:
# #                 # Normal paragraph handling
# #                 if current_tokens + para_tokens <= max_tokens:
# #                     current_chunk += "\n\n" + para
# #                     current_tokens += para_tokens
# #                 else:
# #                     chunks.append(current_chunk.strip())

# #                     # overlap
# #                     overlap_text = get_overlap_text(current_chunk, overlap_tokens)#getoverlaodtext split the text in current chunk to small chunk of overlaping token  limit and combine them into a string 
# #                     current_chunk = overlap_text + "\n\n" + para #scoveerting them back to simple new para and adding the new para to the current chunk
# #                     current_tokens = count_tokens(current_chunk)#counting tokens in the new current chunk to update the token count for the next iteration

# #         if current_chunk:
# #             chunks.append(current_chunk.strip())

# #     return chunks
# # def get_overlap_text(text: str, overlap_tokens: int) -> str:
# #     words = text.split()
# #     return " ".join(words[-overlap_tokens:])



# # if __name__ == "__main__":
# #     sample_text = """
# # # Introduction
# # This is a long document. It contains multiple sections.

# # ## Section One
# # This is paragraph one. It explains something important.

# # This is paragraph two. It continues the explanation. It has multiple sentences.

# # ## Section Two
# # Another section starts here. It also has useful information.

# # """

# #     chunks = build_chunks(sample_text, max_tokens=500, overlap_ratio=0.12)

# #     for i, chunk in enumerate(chunks):
# #         print(f"\n--- Chunk {i+1} ({count_tokens(chunk)} tokens) ---\n")
# #         print(chunk)

# """
# recursive_chunking.py (UPDATED with LangChain)

# - Uses RecursiveCharacterTextSplitter
# - Preserves structure better than manual splitting
# - Supports overlap + hierarchy-aware splitting
# """

# from typing import List

# # LangChain splitter

# from langchain_text_splitters import RecursiveCharacterTextSplitter



# # -----------------------------
# # CLEAN TEXT (optional but useful)
# # -----------------------------
# def clean_text(text: str) -> str:
#     return text.strip()


# # -----------------------------
# # BUILD CHUNKS (LangChain version)
# # -----------------------------
# def build_chunks(
#     text: str,
#     max_tokens: int = 500,
#     overlap_ratio: float = 0.12
# ) -> List[str]:
#     """
#     Recursive chunking using LangChain
#     """

#     text = clean_text(text)

#     chunk_overlap = int(max_tokens * overlap_ratio)

#     # Recursive splitter hierarchy
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=max_tokens,
#         chunk_overlap=chunk_overlap,
#         separators=[
#             "\n\n",   # paragraphs
#             "\n",     # lines
#             ". ",     # sentences
#             " ",      # words
#             ""        # characters (fallback)
#         ]
#     )

#     chunks = splitter.split_text(text)

#     return chunks


# # -----------------------------
# # OPTIONAL: WITH METADATA (if needed later)
# # -----------------------------
# def build_chunks_with_metadata(
#     text: str,
#     source: str = "pdf_1",
#     max_tokens: int = 500,
#     overlap_ratio: float = 0.12
# ):
#     """
#     Returns chunks with basic metadata (LangChain Documents)
#     """

#     from langchain_core.documents import Document
#     chunk_overlap = int(max_tokens * overlap_ratio)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=max_tokens,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     docs = splitter.create_documents([text])

#     # Add metadata
#     for i, doc in enumerate(docs):
#         doc.metadata["chunk_id"] = i
#         doc.metadata["source"] = source

#     return docs


# # -----------------------------
# # TEST
# # -----------------------------
# if __name__ == "__main__":
#     sample_text = """
# # Introduction
# This is a long document. It contains multiple sections.

# ## Section One
# This is paragraph one. It explains something important.

# This is paragraph two. It continues the explanation.

# ## Section Two
# Another section starts here. It also has useful information.
# """

#     chunks = build_chunks(sample_text)

#     for i, c in enumerate(chunks):
#         print(f"\n--- Chunk {i+1} ---\n")
#         print(c)


"""
recursive_chunking.py

Universal Adaptive Recursive Chunking for RAG systems
Supports: PDFs, raw text, code, logs, mixed documents

Author: HelioX Pipeline
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------
# 🔍 Step 1: Detect Document Type
# ---------------------------------------------------

def detect_doc_type(text: str) -> str:
    """
    Detects the type of document based on content.
    """

    text_sample = text[:1000]  # analyze only first part

    if "def " in text_sample or "class " in text_sample:
        return "code"
    elif "|" in text_sample and "-" in text_sample:
        return "table"
    elif "\n\n" in text_sample:
        return "paragraph"
    else:
        return "raw"


# ---------------------------------------------------
# ⚙️ Step 2: Get Adaptive Splitter
# ---------------------------------------------------

def get_splitter(doc_type: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter based on document type.
    """

    if doc_type == "code":
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    elif doc_type == "table":
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n", " ", ""]
        )

    else:  # paragraph / raw / mixed
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",   # paragraphs
                "\n",     # lines
                ". ",     # sentences
                " ",      # words
                ""        # fallback
            ]
        )


# ---------------------------------------------------
# ✂️ Step 3: Base Recursive Chunking
# ---------------------------------------------------

def recursive_chunk(text: str) -> List[str]:
    """
    Performs adaptive recursive chunking.
    """

    doc_type = detect_doc_type(text)
    splitter = get_splitter(doc_type)

    chunks = splitter.split_text(text)
    return chunks


# ---------------------------------------------------
# 🔄 Step 4: Refinement (Optional but recommended)
# ---------------------------------------------------

def refine_chunks(chunks: List[str]) -> List[str]:
    """
    Further splits oversized chunks.
    """

    refined_chunks = []

    for chunk in chunks:
        if len(chunk) > 800:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80
            )
            sub_chunks = sub_splitter.split_text(chunk)
            refined_chunks.extend(sub_chunks)
        else:
            refined_chunks.append(chunk)

    return refined_chunks
 
def build_chunks(text: str) -> List[str]:
    """
    Main function to build chunks from text.
    """

    base_chunks = recursive_chunk(text)
    final_chunks = refine_chunks(base_chunks)

    return final_chunks



if __name__ == "__main__":
    import text_processing
    import os
    
    # Test with sample text first
#     sample_text = """
# # Introduction
# This is a long document. It contains multiple sections.

# ## Section One
# This is paragraph one. It explains something important.

# This is paragraph two. It continues the explanation. It has multiple sentences.

# ## Section Two
# Another section starts here. It also has useful information.
# """

    # chunks = build_chunks(sample_text, max_tokens=500, overlap_ratio=0.12)

    # for i, chunk in enumerate(chunks):
    #     print(f"\n--- Chunk {i+1} ({count_tokens(chunk)} tokens) ---\n")
    #     print(chunk)
    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
    if os.path.exists(folder_path):
        docs = text_processing.load_folder(folder_path)
        if docs:
            print(f"\n\n=== Testing with PDF Files ({len(docs)} documents) ===")
            for doc in docs:
                print(f"\n--- Processing: {doc['metadata']['file_name']} ---")
                chunks = build_chunks(doc['text'])
                print(f"Generated {len(chunks)} chunks")
                if chunks:
                    for i in range(20):  # Print first 3 chunks as a sample
                        print(f"\n--- Sample Chunk {i + 1} ---\n")
                        print(chunks[i][:200] + "...")  # Print first 200 chars of each chunk
                    print(f"Sample Chunk:\n{chunks[1][:200]}...")