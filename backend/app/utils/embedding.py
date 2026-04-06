import uuid
import json
import logging
from typing import List, Any, Dict, Optional

from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk

# Redis
import redis

# NER
import spacy

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "https://b34a5efd-00f2-4e1e-ba1a-b95bd5ec9e77.sa-east-1-0.aws.cloud.qdrant.io",
        qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Mzk3ZTg1ODctNGI3Ny00MDllLTk3MzItN2QxMjc2YzdmZjM0In0.BOctidfeKer1TQ-ENRDDquKSv9jlv8Kilb6FaEiqLoA",
        es_host: str = "https://localhost:9200",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        embedding_batch_size: int = 64,
        qdrant_batch_size: int = 100,
    ):
        self.embedding_batch_size = embedding_batch_size
        self.qdrant_batch_size = qdrant_batch_size

        # ── Embedding model ───────────────────────────────────────────────
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

        # ── Qdrant Cloud ──────────────────────────────────────────────────
        logger.info("Connecting to Qdrant Cloud: %s", qdrant_url)
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = "rag_chunks"
        self._init_qdrant()

        # ── Elasticsearch ─────────────────────────────────────────────────
        # self.es = Elasticsearch(es_host)
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(
            es_host,
            basic_auth=("elastic", "CjRL0vg1D2yulxlEAKWk"),
            verify_certs=False
        )
        self.index_name = "rag_bm25"
        self._init_elasticsearch()

        # ── Redis ─────────────────────────────────────────────────────────
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self._check_redis()

        # ── spaCy NER ─────────────────────────────────────────────────────
        logger.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

    # ──────────────────────────────────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────────────────────────────────

    def _init_qdrant(self) -> None:
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if self.collection_name not in existing:
            logger.info("Creating Qdrant collection '%s'", self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info("Qdrant collection '%s' already exists", self.collection_name)

    def _init_elasticsearch(self) -> None:
        if not self.es.indices.exists(index=self.index_name):
            logger.info("Creating Elasticsearch index '%s'", self.index_name)
            self.es.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        # BM25 is ES default (BM25 similarity); explicit here for clarity
                        "similarity": {"default": {"type": "BM25"}},
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    },
                    "mappings": {
                        "properties": {
                            # full-text BM25 field
                            "text": {"type": "text", "similarity": "BM25"},
                            # multi-value keyword field for NER entities
                            "entities": {"type": "keyword"},
                            # back-reference to the canonical chunk ID
                            "chunk_id": {"type": "keyword"},
                        }
                    },
                },
            )
        else:
            logger.info("Elasticsearch index '%s' already exists", self.index_name)

    def _check_redis(self) -> None:
        try:
            self.redis.ping()
            logger.info("Redis connection OK")
        except redis.ConnectionError as exc:
            raise RuntimeError("Cannot connect to Redis") from exc

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_metadata(metadata: Any) -> Dict:
        """Ensure metadata is JSON-serialisable (drop non-serialisable values)."""
        clean: Dict = {}
        for k, v in (metadata or {}).items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                clean[k] = str(v)
        return clean

    def _extract_entities(self, text: str) -> List[str]:
        """Run spaCy NER and return deduplicated entity strings."""
        doc = self.nlp(text)
        return list({ent.text.strip() for ent in doc.ents if ent.text.strip()})

    def _build_summary(self, text: str) -> str:
        """
        Lightweight extractive summary — first sentence, capped at 200 chars.
        Swap this method body for an LLM call without changing any other code.
        """
        first_sentence = text.split(".")[0].strip()
        return first_sentence[:200] if first_sentence else text[:200]

    def _build_constraints(self, text: str) -> Dict:
        return {
            "char_length": len(text),
            "word_count": len(text.split()),
            "has_numbers": any(c.isdigit() for c in text),
            "has_urls": "http" in text or "www." in text,
        }

    # ──────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def process_chunks(self, chunks: List[Any]) -> None:
        if not chunks:
            logger.warning("No chunks provided — nothing to process.")
            return

        texts: List[str] = [chunk.page_content for chunk in chunks]
        total = len(texts)
        logger.info("Processing %d chunks...", total)

        # ── 1. Batch embeddings ───────────────────────────────────────────
        logger.info("Generating embeddings (batch_size=%d)...", self.embedding_batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
        )

        # ── 2. Build records ──────────────────────────────────────────────
        qdrant_points: List[PointStruct] = []
        es_actions: List[Dict] = []
        redis_pipeline = self.redis.pipeline(transaction=False)

        for i, chunk in enumerate(chunks):
            # Stable UUID per chunk (str for ES/Redis, UUID obj for Qdrant)
            chunk_uuid = uuid.uuid4()
            chunk_id_str = str(chunk_uuid)
            text = chunk.page_content
            metadata = self._safe_metadata(getattr(chunk, "metadata", {}))

            entities = self._extract_entities(text)
            summary = self._build_summary(text)
            constraints = self._build_constraints(text)

            # ── Qdrant point ──────────────────────────────────────────────
            # Entities are also stored here so hybrid re-rankers can use them
            qdrant_points.append(
                PointStruct(
                    id=chunk_id_str,          # Qdrant accepts UUID strings
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": text,
                        "entities": entities,
                        "metadata": metadata,
                    },
                )
            )

            # ── Elasticsearch BM25 action ─────────────────────────────────
            es_actions.append(
                {
                    "_index": self.index_name,
                    "_id": chunk_id_str,
                    "_source": {
                        "text": text,
                        "entities": entities,   # keyword field for exact NER filter
                        "chunk_id": chunk_id_str,
                    },
                }
            )

            # ── Redis chunk profile (pipeline for batched writes) ─────────
            profile = {
                "summary": summary,
                "entities": entities,
                "constraints": constraints,
                "metadata": metadata,
            }
            redis_pipeline.set(f"chunk:{chunk_id_str}", json.dumps(profile))

        # ── 3. Upload to Qdrant (batched) ─────────────────────────────────
        logger.info("Uploading %d points to Qdrant...", len(qdrant_points))
        for start in range(0, len(qdrant_points), self.qdrant_batch_size):
            batch = qdrant_points[start : start + self.qdrant_batch_size]
            self.qdrant.upsert(collection_name=self.collection_name, points=batch)
            logger.info(
                "  Qdrant: upserted %d/%d", min(start + self.qdrant_batch_size, total), total
            )

        # ── 4. Upload to Elasticsearch (bulk) ────────────────────────────
        logger.info("Bulk-indexing %d documents into Elasticsearch...", len(es_actions))
        success, errors = es_bulk(self.es, es_actions, raise_on_error=False)
        if errors:
            logger.warning("Elasticsearch bulk errors: %s", errors)
        else:
            logger.info("  Elasticsearch: indexed %d docs successfully", success)

        # ── 5. Flush Redis pipeline ───────────────────────────────────────
        logger.info("Flushing %d profiles to Redis...", len(qdrant_points))
        redis_pipeline.execute()

        logger.info("✅ Pipeline complete — %d chunks processed.", total)

    # ──────────────────────────────────────────────────────────────────────
    # RETRIEVAL HELPERS (bonus — useful for querying later)
    # ──────────────────────────────────────────────────────────────────────

    def get_chunk_profile(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a chunk profile from Redis by chunk ID."""
        raw = self.redis.get(f"chunk:{chunk_id}")
        return json.loads(raw) if raw else None

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Dense vector search via Qdrant."""
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k
        )
        return [
            {"chunk_id": hit.id, "score": hit.score, **hit.payload}
            for hit in results.points
        ]

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        """Sparse BM25 search via Elasticsearch."""
        resp = self.es.search(
            index=self.index_name,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
        return [
            {
                "chunk_id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"],
            }
            for hit in resp["hits"]["hits"]
        ]


# ──────────────────────────────────────────────────────────────────────────
# USAGE
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document
    from recursive_chunking import build_chunks
    from text_processing import load_folder
    import os

    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")

    # chunks = [
    #     Document(
    #         page_content="OpenAI develops artificial intelligence technologies.",
    #         metadata={"source": "doc1"},
    #     ),
    #     Document(
    #         page_content="Elon Musk founded SpaceX and Tesla.",
    #         metadata={"source": "doc2"},
    #     ),
    # ]
    docs = load_folder(folder_path)
    chunks = build_chunks(docs)

    pipeline = RAGPipeline()  # uses cloud defaults baked into constructor
    pipeline.process_chunks(chunks)

    # Example queries
    print("\n--- Dense search ---")
    for r in pipeline.search_similar("karthik", top_k=2):
        print(r)

    print("\n--- BM25 search ---")
    for r in pipeline.search_bm25("HelioX", top_k=2):
        print(r)

    print("\n--- Chunk profile from Redis ---")
    # Grab first chunk_id from Qdrant to demo profile retrieval
    scroll_result, _ = pipeline.qdrant.scroll(
        collection_name=pipeline.collection_name, limit=1
    )
    if scroll_result:
        cid = str(scroll_result[0].id)
        print(pipeline.get_chunk_profile(cid))