from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentence(text:str) -> List[str]:
    # Split text into sentences based on punctuation
    # 
    return re.split(r'(?<=[.!?])\s+', text.strip())

def count_tokens(text: str) -> int:
    return len(text.split())

def semantic_refine_chunks(
    chunks: List[str],
    similarity_threshold: float = 0.65,
    max_tokens: int = 500,
    overlap_ratio: float = 0.12
) -> List[str]:
    """
    Refines Stage 1 chunks using semantic similarity
    """

    refined_chunks = []
    overlap_tokens = int(max_tokens * overlap_ratio)

    for chunk in chunks:
        sentences = split_sentence(chunk)#splitting the chunk into sentences to calculate the semantic similarity between them and split the chunk based on the similarity threshold

        if len(sentences) <= 1:#we initial need somthing to compare know thats y we use this 
            refined_chunks.append(chunk)#if there is only one sentence in the chunk then we will add it to the refined chunks without any processing
            continue
        # Generate embeddings
        embeddings = model.encode(sentences)

        current_chunk = sentences[0]
        current_tokens = count_tokens(current_chunk)
        for i in range(1, len(sentences)):#### very important concept of cosine similarity calculation to check the semantic similarity between two sentences, if the similarity is less than the threshold then we will split the chunk and add the sentence to the new chunk, otherwise we will add the sentence to the current chunk
            sim = cosine_similarity(
                [embeddings[i - 1]],
                [embeddings[i]]
            )[0][0]

            sentence = sentences[i]
            sent_tokens = count_tokens(sentence)

            # If semantic break OR size overflow → split
            if sim < similarity_threshold or current_tokens + sent_tokens > max_tokens:
                refined_chunks.append(current_chunk.strip())

                # overlap handling
                overlap_text = get_overlap_text(current_chunk, overlap_tokens)#getoverlaodtext split the text in current chunk to small chunk of overlaping token  limit and combine them into a string

                current_chunk = overlap_text + " " + sentence
                current_tokens = count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence
                current_tokens += sent_tokens

        if current_chunk:
            refined_chunks.append(current_chunk.strip())#removing extra whitespace and adding the last chunk to the refined chunks

    return refined_chunks


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    words = text.split()
    return " ".join(words[-overlap_tokens:])


if __name__ == "__main__":
    # Assume stage1_chunks comes from Stage 1
    stage1_chunks = [
        "AI is transforming industries. Machine learning is a subset of AI. It uses data to learn patterns. "
        "Bananas are yellow. Apples are red. Fruits are healthy."
    ]

    refined = semantic_refine_chunks(stage1_chunks, similarity_threshold=0.65)

    for i, chunk in enumerate(refined):#iterating a function so that current chunk can be printed with its index and the refined chunk can be printed in a readable format
        print(f"\n--- Refined Chunk {i+1} ---\n")
        print(chunk)