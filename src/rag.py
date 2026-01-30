"""
RAG module: Hybrid search (Chroma vector + BM25 keyword) with optional metadata filters.
Used as a tool by the chatbot (function calling).
"""
import pickle
import re
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, BM25_INDEX_PATH


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(name=COLLECTION_NAME)


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except (OSError, Exception):
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _norm(s: str) -> str:
    """Default normalized form for matching (stable across any casing)."""
    return (s or "").strip().lower()


def _norm_company(s: str) -> str:
    """Normalized form for company name matching (case-insensitive, spaces removed)."""
    return _norm(s).replace(" ", "")


def _tokenize(text: str) -> list[str]:
    """Same tokenizer as ingest (for BM25)."""
    return re.findall(r"\w+", _norm(text))


_bm25_state = None


def _get_bm25():
    """Load BM25 index and id list from disk."""
    global _bm25_state
    if _bm25_state is None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. Run ingest.py first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_state = pickle.load(f)
    return _bm25_state["index"], _bm25_state["ids"]


def _metadata_matches(meta: dict, primary_sector: Optional[str], location: Optional[str],
                      country: Optional[str], company_name: Optional[str], stage: Optional[str]) -> bool:
    """Check if document metadata matches all non-empty filter values (normalized match)."""
    if primary_sector and primary_sector.strip():
        if (meta.get("primary_sector_n") or _norm(meta.get("primary_sector", ""))) != _norm(primary_sector):
            return False
    if location and location.strip():
        if (meta.get("location_n") or _norm(meta.get("location", ""))) != _norm(location):
            return False
    if country and country.strip():
        if (meta.get("country_n") or _norm(meta.get("country", ""))) != _norm(country):
            return False
    if company_name and company_name.strip():
        meta_n = meta.get("company_n") or _norm(meta.get("company", ""))
        meta_n_no_space = (meta.get("company_n_no_space") or meta_n.replace(" ", ""))
        if meta_n_no_space != _norm_company(company_name):
            return False
    if stage and stage.strip():
        if (meta.get("stage_n") or _norm(meta.get("stage", ""))) != _norm(stage):
            return False
    return True


def _vector_search(
    collection,
    query_embedding: list,
    where: Optional[dict],
    n_results: int,
) -> list[dict]:
    """Run Chroma vector query; return list of {company, content, metadata}."""
    kwargs = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    out = []
    if result["ids"] and result["ids"][0]:
        for i, doc_id in enumerate(result["ids"][0]):
            meta = (result["metadatas"][0][i] or {})
            doc = (result["documents"][0][i] or "")
            out.append({
                "company": meta.get("company", ""),
                "content": doc,
                "metadata": meta,
                "id": doc_id,
            })
    return out


def _bm25_search(
    query_text: str,
    primary_sector: Optional[str],
    location: Optional[str],
    country: Optional[str],
    company_name: Optional[str],
    stage: Optional[str],
    n_results: int,
    id_to_doc: dict,
) -> list[dict]:
    """Run BM25 keyword search, apply metadata filter, return list of {company, content, metadata}."""
    bm25, ids = _get_bm25()
    search_text = query_text.strip() if query_text else "unicorn startup"
    query_tokens = _tokenize(search_text)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for idx in indices:
        if scores[idx] <= 0:
            break
        doc_id = ids[idx]
        if doc_id not in id_to_doc:
            continue
        rec = id_to_doc[doc_id]
        meta = rec.get("metadata", {})
        if not _metadata_matches(meta, primary_sector, location, country, company_name, stage):
            continue
        out.append({
            "company": rec.get("company", ""),
            "content": rec.get("content", ""),
            "metadata": meta,
            "id": doc_id,
        })
        if len(out) >= n_results:
            break
    return out


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: merge two ranked lists by doc id, sort by RRF score."""
    def rrf_score(rank: int) -> float:
        return 1.0 / (k + rank)

    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results, start=1):
        doc_id = r.get("id", "")
        if doc_id:
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank)
    for rank, r in enumerate(bm25_results, start=1):
        doc_id = r.get("id", "")
        if doc_id:
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank)

    id_to_item = {r.get("id"): r for r in vector_results + bm25_results if r.get("id")}
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [id_to_item[doc_id] for doc_id in sorted_ids if doc_id in id_to_item]


def search_companies(
    query_text: str,
    *,
    primary_sector: Optional[str] = None,
    location: Optional[str] = None,
    country: Optional[str] = None,
    company_name: Optional[str] = None,
    stage: Optional[str] = None,
    n_results: int = 10,
) -> list[dict]:
    """
    Hybrid search: vector (Chroma) + BM25 keyword, merged with RRF.
    Optional metadata filters apply to both branches. Returns list of {company, content, metadata}.
    """
    collection = get_collection()
    model = get_embedding_model()
    search_text = query_text.strip() if query_text else "unicorn startup company"
    query_embedding = model.encode([search_text]).tolist()

    where = None
    if primary_sector or location or country or company_name or stage:
        clauses = []
        if primary_sector and primary_sector.strip():
            clauses.append({"primary_sector_n": {"$eq": _norm(primary_sector)}})
        if location and location.strip():
            clauses.append({"location_n": {"$eq": _norm(location)}})
        if country and country.strip():
            clauses.append({"country_n": {"$eq": _norm(country)}})
        if company_name and company_name.strip():
            company_key = "company_n_no_space"
            company_val = _norm_company(company_name)
            clauses.append({company_key: {"$eq": company_val}})
        if stage and stage.strip():
            clauses.append({"stage_n": {"$eq": _norm(stage)}})
        if clauses:
            where = {"$and": clauses} if len(clauses) > 1 else clauses[0]

    # 1) Vector search
    vector_results = _vector_search(collection, query_embedding, where, n_results=n_results)

    # 2) BM25 search (if index exists); otherwise vector-only
    bm25_results = []
    bm25_ids = []
    id_to_doc = {}
    try:
        _, bm25_ids = _get_bm25()
        if bm25_ids:
            get_result = collection.get(ids=bm25_ids, include=["documents", "metadatas"])
            id_to_doc = {}
            if get_result["ids"]:
                for i, doc_id in enumerate(get_result["ids"]):
                    meta = (get_result["metadatas"][i] if get_result["metadatas"] else {}) or {}
                    doc = (get_result["documents"][i] if get_result["documents"] else "") or ""
                    id_to_doc[doc_id] = {
                        "company": meta.get("company", ""),
                        "content": doc,
                        "metadata": meta,
                    }
            bm25_results = _bm25_search(
                query_text, primary_sector, location, country, company_name, stage,
                n_results=n_results, id_to_doc=id_to_doc,
            )
    except FileNotFoundError:
        pass

    # 3) RRF merge and take top n_results (or vector-only if no BM25)
    merged = _rrf_merge(vector_results, bm25_results, k=60)
    out = []
    for r in merged[:n_results]:
        out.append({k: v for k, v in r.items() if k != "id"})

    return out


RESULT_COLUMNS = [
    ("rank", "Rank"),
    ("company", "Company"),
    ("company_link", "Company link"),
    ("company_info", "Company info"),
    ("round_led_by", "Round led by"),
    ("company_background", "Company background"),
    ("founded_year", "Founded year"),
    ("location", "Location"),
    ("country", "Country"),
    ("stage", "Stage"),
    ("primary_sector", "Primary sector"),
    ("time_to_unicorn", "Time to unicorn"),
    ("top_investors_1", "Top investors 1"),
    ("top_investors_2", "Top investors 2"),
    ("annual_revenue", "Annual revenue"),
    ("valuation", "Valuation"),
    ("total_funding_till_date", "Total funding till date"),
    ("employee_count", "Employee count"),
    ("latest_funding_round", "Latest funding round"),
]


def format_search_results(results: list[dict], max_chars: int = 12000) -> str:
    """Format search results for LLM: all 19 columns per result, company name once."""
    parts = []
    total = 0
    for r in results:
        meta = r.get("metadata", {})
        lines = []
        for key, label in RESULT_COLUMNS:
            val = meta.get(key, "") or r.get(key, "") or ""
            lines.append(f"{label}: {(val if isinstance(val, str) else str(val)).strip()}")
        block = "---\n" + "\n".join(lines) + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts) if parts else "No matching companies found in the knowledge base."
