"""
Ingestion script: load data/tracxn.csv, embed with a embedding model, store in Chroma,
and build BM25 index for hybrid keyword search. Run once (or when CSV changes): python ingest.py
"""
import re
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from src.config import CSV_PATH, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, BM25_INDEX_PATH


def _safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def build_document_text(row: pd.Series) -> str:
    """Build a single searchable text block for one company."""
    parts = [
        f"Company: {_safe_str(row.get('Company', ''))}",
        f"Background: {_safe_str(row.get('company_background', ''))}",
        f"Unicorn info: {_safe_str(row.get('company_info', ''))}",
        f"Primary sector: {_safe_str(row.get('primary_sector', ''))}",
        f"Location: {_safe_str(row.get('location', ''))}, Country: {_safe_str(row.get('country', ''))}",
        f"Stage: {_safe_str(row.get('stage', ''))}",
        f"Founded: {_safe_str(row.get('founded_year', ''))}",
        f"Round led by: {_safe_str(row.get('round_led_by', ''))}",
        f"Top investors: {_safe_str(row.get('top_investors_1', ''))}, {_safe_str(row.get('top_investors_2', ''))}",
        f"Valuation: {_safe_str(row.get('valuation', ''))}, Funding: {_safe_str(row.get('total_funding_till_date', ''))}",
        f"Employees: {_safe_str(row.get('employee_count', ''))}",
        f"Latest round: {_safe_str(row.get('latest_funding_round', ''))}",
    ]
    return "\n".join(p for p in parts if p.split(":", 1)[-1].strip())


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, alphanumeric tokens."""
    return re.findall(r"\w+", (text or "").lower())


def _norm(s: str) -> str:
    """Default normalized form for matching (stable across any casing)."""
    return (s or "").strip().lower()


def get_metadata(row: pd.Series) -> dict:
    """Metadata for Chroma: all 19 data columns + _n normalized form for filterable fields."""
    company = _safe_str(row.get("Company", ""))
    primary_sector = _safe_str(row.get("primary_sector", ""))
    location = _safe_str(row.get("location", ""))
    country = _safe_str(row.get("country", ""))
    stage = _safe_str(row.get("stage", ""))
    return {
        "rank": _safe_str(row.get("Rank", "")),
        "company": company,
        "company_n": _norm(company),
        "company_n_no_space": _norm(company).replace(" ", ""),
        "company_link": _safe_str(row.get("company_link", "")),
        "company_info": _safe_str(row.get("company_info", "")),
        "round_led_by": _safe_str(row.get("round_led_by", "")),
        "company_background": _safe_str(row.get("company_background", "")),
        "founded_year": _safe_str(row.get("founded_year", "")),
        "location": location,
        "location_n": _norm(location),
        "country": country,
        "country_n": _norm(country),
        "stage": stage,
        "stage_n": _norm(stage),
        "primary_sector": primary_sector,
        "primary_sector_n": _norm(primary_sector),
        "time_to_unicorn": _safe_str(row.get("time_to_unicorn", "")),
        "top_investors_1": _safe_str(row.get("top_investors_1", "")),
        "top_investors_2": _safe_str(row.get("top_investors_2", "")),
        "annual_revenue": _safe_str(row.get("annual_revenue", "")),
        "valuation": _safe_str(row.get("valuation", "")),
        "total_funding_till_date": _safe_str(row.get("total_funding_till_date", "")),
        "employee_count": _safe_str(row.get("employee_count", "")),
        "latest_funding_round": _safe_str(row.get("latest_funding_round", "")),
    }


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print("Loading CSV...")
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(CSV_PATH, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV; tried utf-8, latin-1, cp1252")

    df = df.dropna(how="all")
    if "Company" not in df.columns:
        raise ValueError("Expected 'Company' column in CSV")

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        company = _safe_str(row.get("Company", ""))
        if not company:
            continue
        doc_id = f"company_{idx}_{company.replace(' ', '_')[:50]}"
        text = build_document_text(row)
        meta = get_metadata(row)
        meta = {k: (v if v else "") for k, v in meta.items()}
        documents.append(text)
        metadatas.append(meta)
        ids.append(doc_id)

    print(f"Loaded {len(documents)} companies. Embedding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents, show_progress_bar=True)
    print("Writing to Chroma...")

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Unicorn startups from tracxn dataset"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Chroma: collection '{COLLECTION_NAME}' has {collection.count()} documents.")

    print("Building BM25 index...")
    corpus_tokens = [tokenize_for_bm25(d) for d in documents]
    bm25 = BM25Okapi(corpus_tokens)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": bm25, "ids": ids}, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}.")


if __name__ == "__main__":
    main()
