import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.uri_parser import parse_uri
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI(title="Pathway-Compatible RAG Service", version="1.0.0")

DATA_DIR = Path(os.getenv("PATHWAY_DATA_DIR", "/app/data"))
REQUIRE_TOKEN = os.getenv("PATHWAY_MCP_TOKEN", "")
TOP_K_DEFAULT = int(os.getenv("PATHWAY_TOP_K", "5"))
INDEX_SOURCE = os.getenv("PATHWAY_INDEX_SOURCE", "mongodb")

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("PATHWAY_MONGODB_DB", "")
SCHEMES_COLLECTION = os.getenv("PATHWAY_SCHEMES_COLLECTION", "schemes")
REPORTS_COLLECTION = os.getenv("PATHWAY_REPORTS_COLLECTION", "anonymousreports")

index_source_used = "none"
last_index_error = ""
active_mongodb_db = ""


doc_store: list[dict[str, Any]] = []
vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = None


class RetrieveRequest(BaseModel):
    query: str
    k: int = TOP_K_DEFAULT


class AnswerRequest(BaseModel):
    prompt: str | None = None
    question: str | None = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def _scheme_to_doc(scheme: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Scheme ID: {_safe_text(scheme.get('id'))}",
            f"Name: {_safe_text(scheme.get('name'))}",
            f"Category: {_safe_text(scheme.get('category'))}",
            f"Village: {_safe_text(scheme.get('village'))}",
            f"District: {_safe_text(scheme.get('district'))}",
            f"Status: {_safe_text(scheme.get('status'))}",
            f"Total Budget: {_safe_text(scheme.get('totalBudget', 0))}",
            f"Budget Utilized: {_safe_text(scheme.get('budgetUtilized', 0))}",
            f"Overall Progress: {_safe_text(scheme.get('overallProgress', 0))}%",
            f"Start Date: {_safe_text(scheme.get('startDate'))}",
            f"End Date: {_safe_text(scheme.get('endDate'))}",
            f"Description: {_safe_text(scheme.get('description'))}",
            f"Phases: {_safe_text(scheme.get('phases', []))}",
            f"Vendor Reports: {_safe_text(scheme.get('vendorReports', []))}",
            f"Discrepancies: {_safe_text(scheme.get('discrepancies', []))}",
        ]
    )


def _report_to_doc(report: dict[str, Any]) -> str:
    anonymized = report.get("anonymizedContent") or {}
    location = report.get("location") or {}
    return "\n".join(
        [
            f"Report ID: {_safe_text(report.get('id'))}",
            f"Status: {_safe_text(report.get('status'))}",
            f"Priority: {_safe_text(report.get('priority'))}",
            f"Category: {_safe_text(anonymized.get('problemCategory'))}",
            f"Severity: {_safe_text(anonymized.get('severity'))}",
            f"Title: {_safe_text(anonymized.get('title'))}",
            f"Description: {_safe_text(anonymized.get('description'))}",
            f"Intent: {_safe_text(anonymized.get('extractedIntent'))}",
            f"Area: {_safe_text(location.get('area'))}",
            f"District: {_safe_text(location.get('district'))}",
            f"Credibility: {_safe_text(report.get('credibilityScore', 0))}",
            f"Created At: {_normalize_date(report.get('createdAt'))}",
        ]
    )


def _resolve_db_name() -> str:
    if MONGODB_DB:
        return MONGODB_DB

    if not MONGODB_URI:
        return "villagetwin"

    try:
        parsed = parse_uri(MONGODB_URI)
        db_name = parsed.get("database")
        return db_name or "villagetwin"
    except Exception:
        return "villagetwin"


def _fetch_docs_from_db(db) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for scheme in db[SCHEMES_COLLECTION].find({}, {"_id": 0}):
        doc_id = _safe_text(scheme.get("id") or "unknown")
        docs.append(
            {
                "id": f"scheme_{doc_id}",
                "path": f"mongodb://{db.name}/{SCHEMES_COLLECTION}/{doc_id}",
                "text": _scheme_to_doc(scheme),
            }
        )

    for report in db[REPORTS_COLLECTION].find({}, {"_id": 0}):
        report_id = _safe_text(report.get("id") or "unknown")
        docs.append(
            {
                "id": f"citizen_report_{report_id}",
                "path": f"mongodb://{db.name}/{REPORTS_COLLECTION}/{report_id}",
                "text": _report_to_doc(report),
            }
        )

    return docs


def _check_auth(auth_header: str | None):
    if not REQUIRE_TOKEN:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1]
    if token != REQUIRE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _load_documents_from_mongodb() -> list[dict[str, Any]]:
    global active_mongodb_db
    if not MONGODB_URI:
        return []

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=6000)

    try:
        preferred_db = _resolve_db_name()
        db = client[preferred_db]
        docs = _fetch_docs_from_db(db)
        if docs:
            active_mongodb_db = preferred_db
            return docs

        for db_name in client.list_database_names():
            if db_name in {"admin", "local", "config"}:
                continue
            if db_name == preferred_db:
                continue

            candidate_docs = _fetch_docs_from_db(client[db_name])
            if candidate_docs:
                active_mongodb_db = db_name
                return candidate_docs

        active_mongodb_db = preferred_db
        return []
    finally:
        client.close()


def _load_documents_from_files() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    if DATA_DIR.exists():
        for path in DATA_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        docs.append(
                            {
                                "id": path.stem,
                                "path": str(path).replace("\\", "/"),
                                "text": text,
                            }
                        )
                except Exception:
                    continue

    return docs


def _load_documents() -> None:
    global doc_store, doc_matrix, index_source_used, last_index_error
    docs: list[dict[str, Any]] = []
    last_index_error = ""

    if INDEX_SOURCE in {"mongodb", "auto"}:
        try:
            docs = _load_documents_from_mongodb()
            if docs:
                index_source_used = "mongodb"
        except Exception as exc:
            last_index_error = str(exc)
            docs = []

    if not docs:
        docs = _load_documents_from_files()
        if docs:
            index_source_used = "files"

    # Provide fallback doc to keep API alive even without indexed files.
    if not docs:
        index_source_used = "empty"
        docs = [
            {
                "id": "empty-index",
                "path": "data/empty-index.txt",
                "text": "No documents are indexed yet. Add text files under the mounted /app/data directory.",
            }
        ]

    doc_store = docs
    doc_matrix = vectorizer.fit_transform([d["text"] for d in doc_store])


def _retrieve(query: str, k: int) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_matrix).flatten()
    top_idx = sims.argsort()[::-1][: max(1, min(k, len(doc_store)))]

    result = []
    for i in top_idx:
        result.append(
            {
                "text": doc_store[i]["text"][:1200],
                "score": float(sims[i]),
                "metadata": {"path": doc_store[i]["path"], "doc_id": doc_store[i]["id"]},
            }
        )
    return result


def _answer_with_context(question: str, docs: list[dict[str, Any]]) -> str:
    context = "\n\n".join([f"Source: {d['metadata']['path']}\n{d['text']}" for d in docs[:5]])

    if HF_API_KEY:
        try:
            prompt = (
                "Answer using ONLY the provided context. If context is insufficient, say so briefly.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}"
            )

            if hf_client:
                completion = hf_client.chat_completion(
                    model=HF_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a concise assistant. Use only the provided context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    temperature=0.2,
                )

                if completion and getattr(completion, "choices", None):
                    content = completion.choices[0].message.content
                    if content and content.strip():
                        return content.strip()

            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.2,
                        "return_full_text": False,
                    },
                },
                timeout=30,
            )

            if response.ok:
                payload = response.json()
                if isinstance(payload, list) and payload:
                    generated = payload[0].get("generated_text", "").strip()
                    if generated:
                        return generated
                if isinstance(payload, dict):
                    generated = payload.get("generated_text", "").strip()
                    if generated:
                        return generated
        except Exception:
            pass

    if docs:
        return (
            "Context-based answer (LLM unavailable):\n"
            + "\n".join([f"- {d['text'][:220]}..." for d in docs[:3]])
        )

    return "Insufficient data to answer this question."


@app.on_event("startup")
def startup_event():
    _load_documents()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indexed_docs": len(doc_store),
        "data_dir": str(DATA_DIR),
        "index_source": index_source_used,
        "mongodb_db": active_mongodb_db,
        "last_index_error": last_index_error,
        "llm_provider": "huggingface" if HF_API_KEY else "fallback",
        "llm_model": HF_MODEL,
        "llm_enabled": bool(HF_API_KEY),
    }


@app.post("/v1/retrieve")
def retrieve(payload: RetrieveRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    docs = _retrieve(payload.query, payload.k)
    return docs


@app.post("/v1/pw_ai_answer")
def ai_answer(payload: AnswerRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    question = (payload.prompt or payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="prompt or question is required")

    docs = _retrieve(question, TOP_K_DEFAULT)
    answer = _answer_with_context(question, docs)
    return {"response": answer, "sources": docs}


@app.post("/v1/index/reload")
def reload_index(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    _load_documents()
    return {"status": "reloaded", "indexed_docs": len(doc_store)}
