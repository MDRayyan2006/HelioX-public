# import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
import uuid, json, logging, os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
import psycopg2
from psycopg2.extras import execute_values
import redis
import spacy

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    _UUID_NAMESPACE = uuid.NAMESPACE_URL

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "https://b34a5efd-00f2-4e1e-ba1a-b95bd5ec9e77.sa-east-1-0.aws.cloud.qdrant.io",
        qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Mzk3ZTg1ODctNGI3Ny00MDllLTk3MzItN2QxMjc2YzdmZjM0In0.BOctidfeKer1TQ-ENRDDquKSv9jlv8Kilb6FaEiqLoA",
        es_host: str = "http://localhost:9200",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_db: str = "postgres",
        pg_user: str = "postgres",
        pg_password: str = "1234",
        embedding_batch_size: int = 64,
        qdrant_batch_size: int = 100,
        ner_batch_size: int = 32,        # NEW: batch NER
    ):
        self.embedding_batch_size = embedding_batch_size
        self.qdrant_batch_size = qdrant_batch_size
        self.ner_batch_size = ner_batch_size

        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

        logger.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])  # faster: NER only

        logger.info("Connecting to stores...")
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "rag_chunks"
        self._init_qdrant()

        self.es = Elasticsearch(
            es_host,
            basic_auth=("elastic", "PyWbRta28AJGldbDY7W5"),
            verify_certs=False
        )
        self.index_name = "rag_bm25"
        self._init_elasticsearch()

        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self._check_redis()

        self.pg_conn = psycopg2.connect(
            host=pg_host, port=pg_port, dbname=pg_db,
            user=pg_user, password=pg_password
        )
        self.pg_conn.autocommit = False
        self._init_postgres()

    # ── Init methods (unchanged) ────────────────────────────────────────
    def _init_qdrant(self):
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if self.collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _init_elasticsearch(self):
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body={
                "settings": {
                    "similarity": {"default": {"type": "BM25"}},
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "mappings": {"properties": {
                    "text": {"type": "text", "similarity": "BM25"},
                    "entities": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                }},
            })

    def _check_redis(self):
        try:
            self.redis.ping()
        except redis.ConnectionError as e:
            raise RuntimeError("Cannot connect to Redis") from e

    def _init_postgres(self):
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_chunks (
                    chunk_id   UUID        PRIMARY KEY,
                    text       TEXT        NOT NULL,
                    metadata   JSONB       NOT NULL DEFAULT '{}',
                    source     TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_raw_chunks_source ON raw_chunks (source);
            """)
        self.pg_conn.commit()

    # ── Helpers ─────────────────────────────────────────────────────────
    @classmethod
    def _make_chunk_id(cls, text: str, metadata: Dict) -> str:
        return str(uuid.uuid5(cls._UUID_NAMESPACE, f"{metadata.get('source','')}::{text}"))

    @staticmethod
    def _safe_metadata(metadata: Any) -> Dict:
        clean = {}
        for k, v in (metadata or {}).items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                clean[k] = str(v)
        return clean

    def _batch_extract_entities(self, texts: List[str]) -> List[List[str]]:
        """Run NER on all texts in one batched pass — much faster than one-by-one."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=self.ner_batch_size):
            results.append(list({ent.text.strip() for ent in doc.ents if ent.text.strip()}))
        return results

    def _build_summary(self, text: str) -> str:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        return sentences[0][:200] if sentences else text[:200]

    def _build_constraints(self, text: str) -> Dict:
        return {
            "char_length": len(text),
            "word_count": len(text.split()),
            "has_numbers": any(c.isdigit() for c in text),
            "has_urls": "http" in text or "www." in text,
        }

    # ── Storage writers (called concurrently) ───────────────────────────
    def _write_qdrant(self, points: List[PointStruct], total: int):
        for start in range(0, len(points), self.qdrant_batch_size):
            batch = points[start: start + self.qdrant_batch_size]
            self.qdrant.upsert(collection_name=self.collection_name, points=batch)
            logger.info("Qdrant: upserted %d/%d", min(start + self.qdrant_batch_size, total), total)

    def _write_elasticsearch(self, actions: List[Dict]):
        success, errors = es_bulk(self.es, actions, raise_on_error=False)
        if errors:
            logger.warning("ES bulk errors: %s", errors)
        else:
            logger.info("ES: indexed %d docs", success)

    def _write_redis(self, pipeline):
        pipeline.execute()
        logger.info("Redis: profiles flushed")

    def _write_postgres(self, rows: List[tuple]):
        try:
            with self.pg_conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO raw_chunks (chunk_id, text, metadata, source)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, source=EXCLUDED.source
                """, rows)
            self.pg_conn.commit()
            logger.info("PG: inserted/updated %d rows", len(rows))
        except Exception as e:
            self.pg_conn.rollback()
            logger.error("PG insert failed: %s", e)
            raise

    # ── Main pipeline ────────────────────────────────────────────────────
    def process_chunks(self, chunks: List[Any]) -> None:
        if not chunks:
            logger.warning("No chunks to process.")
            return

        texts = [c.page_content for c in chunks]
        total = len(texts)
        logger.info("Processing %d chunks...", total)

        # 1. Embeddings (batched, fast)
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # 2. NER — ONE batched pass over all texts (not N individual calls)
        logger.info("Extracting entities (batched)...")
        all_entities = self._batch_extract_entities(texts)

        # 3. Build all records
        qdrant_points, es_actions, pg_rows = [], [], []
        redis_pipe = self.redis.pipeline(transaction=False)

        for i, chunk in enumerate(chunks):
            metadata = self._safe_metadata(getattr(chunk, "metadata", {}))
            chunk_id = self._make_chunk_id(chunk.page_content, metadata)
            text = chunk.page_content
            entities = all_entities[i]

            qdrant_points.append(PointStruct(
                id=chunk_id,
                vector=embeddings[i].tolist(),
                payload={"text": text, "entities": entities, "metadata": metadata},
            ))

            es_actions.append({
                "_index": self.index_name,
                "_id": chunk_id,
                "_source": {"text": text, "entities": entities, "chunk_id": chunk_id},
            })

            redis_pipe.set(f"chunk:{chunk_id}", json.dumps({
                "summary": self._build_summary(text),
                "entities": entities,
                "constraints": self._build_constraints(text),
                "metadata": metadata,
            }))

            pg_rows.append((chunk_id, text, json.dumps(metadata), metadata.get("source")))

        # 4. Write to all 4 stores CONCURRENTLY
        logger.info("Writing to all stores concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._write_qdrant, qdrant_points, total): "qdrant",
                executor.submit(self._write_elasticsearch, es_actions): "elasticsearch",
                executor.submit(self._write_redis, redis_pipe): "redis",
                executor.submit(self._write_postgres, pg_rows): "postgres",
            }
            for future in concurrent.futures.as_completed(futures):
                store = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("Store %s failed: %s", store, e)

        logger.info("Pipeline complete — %d chunks processed.", total)

    # ── Retrieval ────────────────────────────────────────────────────────
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name, query=vector, limit=top_k
        )
        return [{"chunk_id": h.id, "score": h.score, **h.payload} for h in results.points]

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        resp = self.es.search(
            index=self.index_name,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        return [{"chunk_id": h["_id"], "score": h["_score"], **h["_source"]}
                for h in resp["hits"]["hits"]]

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search: merge dense + BM25 results using Reciprocal Rank Fusion.
        alpha controls dense weight (1.0 = pure dense, 0.0 = pure BM25)
        """
        dense_results = self.search_similar(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        scores: Dict[str, float] = {}
        k = 60  # RRF constant

        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + alpha * (1 / (k + rank + 1))

        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1 / (k + rank + 1))

        # Merge payloads
        all_docs = {r["chunk_id"]: r for r in dense_results + bm25_results}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"rrf_score": score, **all_docs[cid]} for cid, score in ranked]

    def get_chunk_profile(self, chunk_id: str) -> Optional[Dict]:
        raw = self.redis.get(f"chunk:{chunk_id}")
        return json.loads(raw) if raw else None

    def get_raw_chunk(self, chunk_id: str) -> Optional[Dict]:
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, text, metadata, source, created_at FROM raw_chunks WHERE chunk_id = %s",
                (chunk_id,)
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"chunk_id": str(row[0]), "text": row[1],
                "metadata": row[2], "source": row[3], "created_at": row[4].isoformat()}

    def __del__(self):
        try:
            if self.pg_conn and not self.pg_conn.closed:
                self.pg_conn.close()
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from text_processing import load_folder
    from recursive_chunking import chunk_documents
    import os, time

    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
    docs = load_folder(folder_path)
    chunks = chunk_documents(docs)

    pipeline = RAGPipeline()
    pipeline.process_chunks(chunks)

    print("\n--- Hybrid search (best retrieval) ---")
    for r in pipeline.search_hybrid("projects", top_k=3, alpha=0.6):
        print(f"  [{r['rrf_score']:.4f}] {r.get('text','')}")