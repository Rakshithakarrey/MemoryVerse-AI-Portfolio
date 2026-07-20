"""
================================================================================
 MemoryVerse AI '26 — Digital Portfolio & Memory Engine
================================================================================
A single-file Streamlit application that ingests a student's documents
(resumes, certificates, project reports, internship letters), uses an LLM to
categorize/tag them, builds a skill-relationship graph, renders a visual
timeline of the student's growth, and provides a semantic ("Smart Retrieval")
search engine over the whole portfolio.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
1) Install dependencies (only `streamlit` and `numpy` are hard requirements;
   everything else degrades gracefully if not installed):

    pip install streamlit numpy pandas pillow pdfplumber PyPDF2 openai anthropic pyvis

2) Run:

    streamlit run memoryverse_app.py

3) In the sidebar, choose your LLM provider (OpenAI or Anthropic) and paste
   your API key. Nothing is sent anywhere except directly to that provider's
   official API — the key is only kept in Streamlit's session memory.

--------------------------------------------------------------------------
ARCHITECTURE NOTES (for the Thought Process Sheet)
--------------------------------------------------------------------------
- Storage: SQLite (`memoryverse.db`) holds structured metadata. Original
  uploaded files are copied to a local `mv_storage/` folder so they can be
  re-downloaded/viewed later.
- Schema: the brief specifies `documents` and `relationships` tables. We
  honor that, and add a small `skills` lookup table + `source_type` /
  `target_type` columns on `relationships` so a relationship edge can point
  at either a document node or a skill node (needed to build the bipartite
  Certification → Skill → Project → Internship graph in Module 3).
- Ingestion (Module 1 & 2): text is pulled out of PDFs (pdfplumber, falling
  back to PyPDF2) and TXT files directly. Images (and PDFs with no
  extractable text, e.g. scans) are instead sent straight to the LLM's
  *vision* endpoint — this avoids needing an OCR engine installed locally
  and is more robust than OCR for messy certificate scans.
- Structured Output: the LLM is instructed (system prompt + JSON mode where
  available) to return ONLY a JSON object matching a fixed schema. We
  defensively strip markdown fences and re-attempt parsing before failing.
- RAG (Module 5): every document is embedded with a deterministic, dependency
  -free hashing-vectorizer (bag-of-words hashed into a fixed-size vector,
  L2-normalized) so semantic search works out of the box with zero extra
  installs. If the user supplies an OpenAI key, they can flip a toggle to
  use real `text-embedding-3-small` embeddings for higher-quality retrieval.
  Cosine similarity ranks documents against the query vector — this is the
  same core idea as a vector DB (like ChromaDB), just implemented directly
  with numpy so the whole app stays a single file with no external services.
================================================================================
"""

import streamlit as st
import sqlite3
import json
import os
import io
import re
import base64
import hashlib
import uuid
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Optional dependencies — the app must never crash on import. Each capability
# is feature-flagged and the UI explains what to `pip install` if missing.
# ----------------------------------------------------------------------------
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

import streamlit.components.v1 as components


# ==============================================================================
# CONSTANTS
# ==============================================================================
DB_PATH = "memoryverse.db"
STORAGE_DIR = "mv_storage"

CATEGORIES = [
    "Projects", "Skills", "Certifications",
    "Internships", "Achievements", "Academics",
]

CATEGORY_COLORS = {
    "Projects":       "#6C5CE7",
    "Skills":         "#00B894",
    "Certifications": "#0984E3",
    "Internships":    "#E17055",
    "Achievements":   "#FDCB6E",
    "Academics":      "#D63031",
}

CATEGORY_ICONS = {
    "Projects": "🛠️", "Skills": "⚡", "Certifications": "📜",
    "Internships": "💼", "Achievements": "🏆", "Academics": "🎓",
}

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
EMBED_DIM = 512  # dimensionality of the local hashing-vectorizer embeddings

SYSTEM_PROMPT = """You are the ingestion engine for MemoryVerse, a student's AI-powered digital portfolio.
You will be given the extracted text (and/or an image) of ONE uploaded document, which could be a resume,
certificate, project report, internship/offer letter, marksheet, or achievement certificate.

Analyze it carefully and respond with ONLY a single valid JSON object — no markdown fences, no commentary,
no explanation before or after — matching EXACTLY this schema:

{
  "category": one of ["Projects", "Skills", "Certifications", "Internships", "Achievements", "Academics"],
  "title": "short human-readable name of the document/milestone",
  "date": "YYYY-MM-DD if a full date is present, otherwise YYYY, otherwise your best estimate YYYY",
  "extracted_skills": ["skill1", "skill2", "..."],
  "summary": "one or two sentence plain-English summary of what this document represents"
}

Rules:
- "extracted_skills" should be concise, deduplicated, title-cased technical/soft skills (e.g. "Python",
  "Public Speaking", "React", "SQL", "Team Leadership") — 0 to 10 items. If truly none are relevant, use [].
- Pick the SINGLE best-fitting category even if the document could arguably fit two.
- If no date is discoverable, use your best reasonable estimate rather than leaving it blank.
- Never wrap the JSON in ```json fences. Output raw JSON only.
"""


# ==============================================================================
# DATABASE LAYER
# ==============================================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't already exist. Safe to call every run."""
    conn = get_conn()
    cur = conn.cursor()

    # --- Required schema: documents -----------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            category        TEXT NOT NULL,
            extracted_text  TEXT,
            metadata_json   TEXT,
            date            TEXT,
            storage_path    TEXT
        )
    """)

    # --- Lookup table for skill nodes (extension for the graph engine) ------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT UNIQUE NOT NULL
        )
    """)

    # --- Required schema: relationships (extended with *_type so an edge ----
    #     can point at either a 'document' node or a 'skill' node) -----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id   INTEGER NOT NULL,
            target_id   INTEGER NOT NULL,
            rel_type    TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'document',
            target_type TEXT NOT NULL DEFAULT 'document'
        )
    """)

    conn.commit()
    conn.close()


def reset_db():
    """Wipe all data (tables + stored files) and re-initialize a clean DB."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("DROP TABLE IF EXISTS skills")
    cur.execute("DROP TABLE IF EXISTS relationships")
    conn.commit()
    conn.close()

    if os.path.isdir(STORAGE_DIR):
        for f in os.listdir(STORAGE_DIR):
            try:
                os.remove(os.path.join(STORAGE_DIR, f))
            except OSError:
                pass

    init_db()


def insert_document(filename, category, extracted_text, metadata_dict, date_str, storage_path):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO documents (filename, category, extracted_text, metadata_json, date, storage_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (filename, category, extracted_text, json.dumps(metadata_dict), date_str, storage_path),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def get_or_create_skill(skill_name):
    skill_name = skill_name.strip()
    if not skill_name:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM skills WHERE name = ?", (skill_name,))
    row = cur.fetchone()
    if row:
        skill_id = row["id"]
    else:
        cur.execute("INSERT INTO skills (name) VALUES (?)", (skill_name,))
        conn.commit()
        skill_id = cur.lastrowid
    conn.close()
    return skill_id


def link_document_to_skill(doc_id, skill_id):
    """Module 3: automatically link a processed document to each extracted skill."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO relationships (source_id, target_id, rel_type, source_type, target_type)
           VALUES (?, ?, 'has_skill', 'document', 'skill')""",
        (doc_id, skill_id),
    )
    conn.commit()
    conn.close()


def fetch_all_documents():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM documents ORDER BY id DESC", conn)
    conn.close()
    return df


def fetch_all_skills():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM skills ORDER BY name", conn)
    conn.close()
    return df


def fetch_all_relationships():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM relationships", conn)
    conn.close()
    return df


def db_stats():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM documents")
    n_docs = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM skills")
    n_skills = cur.fetchone()["c"]
    cur.execute("SELECT category, COUNT(*) AS c FROM documents GROUP BY category")
    by_cat = {r["category"]: r["c"] for r in cur.fetchall()}
    conn.close()
    return n_docs, n_skills, by_cat


# ==============================================================================
# FILE HANDLING & TEXT EXTRACTION  (Module 1)
# ==============================================================================
def ensure_storage_dir():
    Path(STORAGE_DIR).mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    """Persist the raw bytes to disk under a unique name; return the path."""
    ensure_storage_dir()
    unique_name = f"{uuid.uuid4().hex[:10]}_{uploaded_file.name}"
    path = os.path.join(STORAGE_DIR, unique_name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Try pdfplumber first (better layout handling), fall back to PyPDF2."""
    text_chunks = []

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
            if text_chunks:
                return "\n".join(text_chunks)
        except Exception:
            pass  # fall through to PyPDF2

    if PYPDF2_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
            if text_chunks:
                return "\n".join(text_chunks)
        except Exception:
            pass

    return ""  # scanned/unreadable PDF — caller will fall back to vision


def get_image_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    return "image/png"


def render_pdf_first_page_as_image(file_bytes: bytes):
    """Best-effort: rasterize a scanned/no-text PDF's first page for vision fallback."""
    if not PDFPLUMBER_AVAILABLE:
        return None
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page = pdf.pages[0]
            pil_img = page.to_image(resolution=150).original
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


# ==============================================================================
# LLM CATEGORIZATION  (Module 2 — Structured Outputs)
# ==============================================================================
def clean_json_response(raw_text: str) -> dict:
    """Defensively strip markdown fences / stray text and parse JSON."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    # If there's leading/trailing prose, grab the outermost { ... } block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def validate_and_normalize(data: dict, fallback_title: str) -> dict:
    category = data.get("category", "Achievements")
    if category not in CATEGORIES:
        category = "Achievements"

    title = str(data.get("title") or fallback_title).strip()[:200]

    date_str = str(data.get("date") or dt.datetime.now().year).strip()
    if not re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", date_str):
        year_match = re.search(r"(19|20)\d{2}", date_str)
        date_str = year_match.group(0) if year_match else str(dt.datetime.now().year)

    skills = data.get("extracted_skills") or []
    if not isinstance(skills, list):
        skills = []
    skills = [str(s).strip().title() for s in skills if str(s).strip()][:10]

    summary = str(data.get("summary") or "").strip()[:500]

    return {
        "category": category,
        "title": title,
        "date": date_str,
        "extracted_skills": skills,
        "summary": summary,
    }


def categorize_with_openai(api_key, model, filename, text_content, image_bytes, image_mime):
    client = OpenAI(api_key=api_key)

    user_content = [{"type": "text", "text": f"Filename: {filename}\n\nExtracted text:\n{text_content[:12000]}"}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{b64}"},
        })

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    raw = response.choices[0].message.content
    return clean_json_response(raw)


def categorize_with_anthropic(api_key, model, filename, text_content, image_bytes, image_mime):
    client = anthropic.Anthropic(api_key=api_key)

    user_content = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": image_mime, "data": b64},
        })
    user_content.append({
        "type": "text",
        "text": f"Filename: {filename}\n\nExtracted text:\n{text_content[:12000]}\n\nRespond with ONLY the JSON object described in your instructions.",
    })

    response = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return clean_json_response(raw)


def categorize_document(provider, api_key, model, filename, text_content, image_bytes, image_mime):
    """
    Dispatches to the selected provider. Always sends *something* meaningful:
    - If there's extracted text, it's sent (plus an image if we have one, e.g. resumes
      that are also images, or a scanned PDF's rasterized first page).
    - If there's no text at all, the image is the sole source of truth (vision-only mode).
    Returns a normalized, schema-validated dict. Raises Exception with a readable
    message on failure so the UI can surface it.
    """
    if not text_content and not image_bytes:
        raise ValueError("No readable text or image content could be extracted from this file.")

    if provider == "OpenAI":
        if not OPENAI_SDK_AVAILABLE:
            raise RuntimeError("The `openai` package isn't installed. Run: pip install openai")
        raw = categorize_with_openai(api_key, model, filename, text_content, image_bytes, image_mime)
    else:
        if not ANTHROPIC_SDK_AVAILABLE:
            raise RuntimeError("The `anthropic` package isn't installed. Run: pip install anthropic")
        raw = categorize_with_anthropic(api_key, model, filename, text_content, image_bytes, image_mime)

    return validate_and_normalize(raw, fallback_title=filename)


# ==============================================================================
# LOCAL EMBEDDINGS + COSINE SIMILARITY  (Module 5 — RAG core)
# ==============================================================================
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def hashing_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Zero-dependency 'embedding': a hashing-trick bag-of-words vector.
    Each token is hashed into one of `dim` buckets and counted (term frequency).
    We then apply a log(1+x) dampening (so very frequent words don't dominate)
    and L2-normalize — this makes cosine similarity behave sensibly, the same
    way TF-IDF-style vectors do in a real vector database. This is the
    "zero-config" substitute for calling an embeddings API.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokenize(text):
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    vec = np.log1p(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def openai_embed(client, text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_document_corpus_text(row) -> str:
    """What we embed for each document: title + summary + skills + a slice of raw text."""
    meta = json.loads(row["metadata_json"] or "{}")
    parts = [
        row["filename"],
        meta.get("title", ""),
        meta.get("summary", ""),
        " ".join(meta.get("extracted_skills", [])),
        row["category"],
        (row["extracted_text"] or "")[:2000],
    ]
    return " \n ".join(p for p in parts if p)


# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="MemoryVerse AI '26", page_icon="🧠", layout="wide")

init_db()
ensure_storage_dir()

# --- Global CSS ---------------------------------------------------------------
st.markdown("""
<style>
    .mv-hero {
        padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #1f1147 0%, #3d1f6b 45%, #6C5CE7 100%);
        color: white;
    }
    .mv-hero h1 { margin: 0; font-size: 1.9rem; }
    .mv-hero p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    .mv-card {
        border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.9rem;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    }
    .mv-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; color: white; letter-spacing: .02em;
    }
    .mv-skill-chip {
        display: inline-block; padding: 0.12rem 0.55rem; margin: 0.15rem 0.25rem 0 0;
        border-radius: 999px; font-size: 0.72rem; background: rgba(108,92,231,0.18);
        border: 1px solid rgba(108,92,231,0.4); color: #cbb9ff;
    }

    /* Timeline */
    .mv-timeline { position: relative; margin-left: 20px; padding-left: 28px;
        border-left: 3px solid rgba(255,255,255,0.15); }
    .mv-timeline-item { position: relative; margin-bottom: 1.4rem; }
    .mv-timeline-dot {
        position: absolute; left: -37.5px; top: 4px; width: 16px; height: 16px;
        border-radius: 50%; border: 3px solid #0e1117;
    }
    .mv-timeline-year {
        font-size: 0.78rem; font-weight: 700; opacity: 0.6; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mv-hero">
  <h1>🧠 MemoryVerse AI</h1>
  <p>Your AI-powered digital portfolio — upload once, and let AI categorize, connect, and narrate your journey.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SIDEBAR — configuration
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"], index=0)

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        help="Stored only in this browser session's memory — never written to disk.",
        key="api_key_input",
    )

    default_model = DEFAULT_OPENAI_MODEL if provider == "OpenAI" else DEFAULT_ANTHROPIC_MODEL
    model_name = st.text_input("Model", value=default_model,
                                help="Override if you want a different model version.")

    sdk_ok = OPENAI_SDK_AVAILABLE if provider == "OpenAI" else ANTHROPIC_SDK_AVAILABLE
    if not sdk_ok:
        pkg = "openai" if provider == "OpenAI" else "anthropic"
        st.warning(f"`{pkg}` isn't installed. Run:\n\n`pip install {pkg}`")

    st.divider()

    use_openai_embeddings = False
    if provider == "OpenAI" and api_key and OPENAI_SDK_AVAILABLE:
        use_openai_embeddings = st.checkbox(
            "🎯 Use real OpenAI embeddings for search",
            value=False,
            help="Off = free, zero-config local hashing-vector search. On = higher quality "
                 "semantic search using text-embedding-3-small (uses your API key).",
        )

    st.divider()
    st.subheader("📊 Portfolio Stats")
    n_docs, n_skills, by_cat = db_stats()
    c1, c2 = st.columns(2)
    c1.metric("Documents", n_docs)
    c2.metric("Unique Skills", n_skills)
    if by_cat:
        st.caption("By category")
        st.dataframe(
            pd.DataFrame({"Category": list(by_cat.keys()), "Count": list(by_cat.values())}),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    st.subheader("🗑️ Danger Zone")
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("Reset Database", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.error("This permanently deletes ALL documents & files. Are you sure?")
        cc1, cc2 = st.columns(2)
        if cc1.button("✅ Yes, wipe it", use_container_width=True):
            reset_db()
            st.session_state.confirm_reset = False
            st.session_state.pop("processed_file_ids", None)
            st.success("Database reset.")
            st.rerun()
        if cc2.button("Cancel", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()


# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab_upload, tab_timeline, tab_graph, tab_search = st.tabs(
    ["📥 Upload & Process", "⏳ My Digital Timeline", "🕸 Knowledge Graph", "🔍 Smart Semantic Search"]
)


# ==============================================================================
# TAB 1 — UPLOAD & PROCESS  (Modules 1, 2, 3)
# ==============================================================================
with tab_upload:
    st.subheader("📥 Upload documents")
    st.caption("Resumes, certificates, project reports, internship/offer letters, marksheets — PDF, TXT, or image.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if "processed_file_ids" not in st.session_state:
        st.session_state.processed_file_ids = set()

    process_clicked = st.button(
        "🚀 Process with AI", type="primary",
        disabled=not uploaded_files, use_container_width=True,
    )

    if process_clicked:
        if not api_key:
            st.error(f"Please enter your {provider} API key in the sidebar first.")
        elif not sdk_ok:
            pkg = "openai" if provider == "OpenAI" else "anthropic"
            st.error(f"Please install the `{pkg}` package first: `pip install {pkg}`")
        else:
            progress = st.progress(0.0, text="Starting...")
            results = []

            for i, uf in enumerate(uploaded_files):
                file_key = f"{uf.name}_{uf.size}"
                progress.progress((i) / len(uploaded_files), text=f"Processing {uf.name}...")

                file_bytes = uf.getvalue()
                ext = uf.name.lower().rsplit(".", 1)[-1]

                text_content = ""
                image_bytes = None
                image_mime = None

                # --- Module 1: extraction -----------------------------------
                if ext == "pdf":
                    text_content = extract_text_from_pdf(file_bytes)
                    if not text_content.strip():
                        # Likely a scanned PDF — fall back to vision on page 1.
                        rasterized = render_pdf_first_page_as_image(file_bytes)
                        if rasterized:
                            image_bytes = rasterized
                            image_mime = "image/png"
                elif ext == "txt":
                    try:
                        text_content = file_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        text_content = ""
                else:  # image
                    image_bytes = file_bytes
                    image_mime = get_image_mime(uf.name)

                # --- Module 2: LLM categorization ---------------------------
                try:
                    result = categorize_document(
                        provider, api_key, model_name, uf.name,
                        text_content, image_bytes, image_mime,
                    )
                    error = None
                except Exception as e:
                    result = None
                    error = str(e)

                if result:
                    storage_path = save_uploaded_file(uf)
                    stored_text = text_content if text_content.strip() else "[No extractable text — analyzed via vision model]"

                    doc_id = insert_document(
                        filename=uf.name,
                        category=result["category"],
                        extracted_text=stored_text,
                        metadata_dict=result,
                        date_str=result["date"],
                        storage_path=storage_path,
                    )

                    # --- Module 3: relationship engine ----------------------
                    for skill_name in result["extracted_skills"]:
                        skill_id = get_or_create_skill(skill_name)
                        if skill_id:
                            link_document_to_skill(doc_id, skill_id)

                    st.session_state.processed_file_ids.add(file_key)

                results.append((uf.name, result, error))

            progress.progress(1.0, text="Done!")
            progress.empty()

            st.success(f"Processed {sum(1 for _, r, _ in results if r)} of {len(results)} file(s).")

            for fname, result, error in results:
                if result:
                    icon = CATEGORY_ICONS.get(result["category"], "📄")
                    color = CATEGORY_COLORS.get(result["category"], "#888")
                    with st.expander(f"{icon} {fname} → **{result['title']}**", expanded=False):
                        b1, b2 = st.columns([1, 3])
                        with b1:
                            st.markdown(
                                f'<span class="mv-badge" style="background:{color}">{result["category"]}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"📅 {result['date']}")
                        with b2:
                            st.write(result["summary"])
                            if result["extracted_skills"]:
                                chips = "".join(
                                    f'<span class="mv-skill-chip">{s}</span>' for s in result["extracted_skills"]
                                )
                                st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.error(f"❌ {fname}: {error}")

    st.divider()
    st.subheader("📁 All Documents in Portfolio")
    docs_df = fetch_all_documents()
    if docs_df.empty:
        st.info("No documents yet. Upload something above to get started!")
    else:
        for _, row in docs_df.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            color = CATEGORY_COLORS.get(row["category"], "#888")
            icon = CATEGORY_ICONS.get(row["category"], "📄")
            with st.container():
                st.markdown('<div class="mv-card">', unsafe_allow_html=True)
                cols = st.columns([5, 2, 2])
                cols[0].markdown(f"**{icon} {meta.get('title', row['filename'])}**")
                cols[0].caption(row["filename"])
                cols[1].markdown(
                    f'<span class="mv-badge" style="background:{color}">{row["category"]}</span>',
                    unsafe_allow_html=True,
                )
                cols[2].caption(f"📅 {row['date']}")
                if os.path.exists(row["storage_path"]):
                    with open(row["storage_path"], "rb") as fh:
                        cols[2].download_button(
                            "⬇️ Download", data=fh.read(), file_name=row["filename"],
                            key=f"dl_list_{row['id']}", use_container_width=True,
                        )
                st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2 — DIGITAL JOURNEY TIMELINE  (Module 4)
# ==============================================================================
with tab_timeline:
    st.subheader("⏳ My Digital Journey")
    docs_df = fetch_all_documents()

    if docs_df.empty:
        st.info("Your timeline will appear here once you've uploaded and processed documents.")
    else:
        def sort_key(date_str):
            m = re.match(r"^(\d{4})(-(\d{2}))?(-(\d{2}))?$", str(date_str))
            if not m:
                return (0, 0, 0)
            y = int(m.group(1))
            mo = int(m.group(3)) if m.group(3) else 1
            d = int(m.group(5)) if m.group(5) else 1
            return (y, mo, d)

        docs_df["_sort"] = docs_df["date"].apply(sort_key)
        docs_df = docs_df.sort_values("_sort", ascending=False)

        filter_cats = st.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)
        docs_df = docs_df[docs_df["category"].isin(filter_cats)]

        current_year = None
        st.markdown('<div class="mv-timeline">', unsafe_allow_html=True)
        for _, row in docs_df.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            year = str(row["date"])[:4]
            color = CATEGORY_COLORS.get(row["category"], "#888")
            icon = CATEGORY_ICONS.get(row["category"], "📄")

            skills_html = "".join(
                f'<span class="mv-skill-chip">{s}</span>' for s in meta.get("extracted_skills", [])
            )

            year_label = f'<div class="mv-timeline-year">{year}</div>' if year != current_year else ""
            current_year = year

            st.markdown(f"""
            <div class="mv-timeline-item">
                <div class="mv-timeline-dot" style="background:{color}"></div>
                {year_label}
                <div class="mv-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>{icon} {meta.get('title', row['filename'])}</strong>
                        <span class="mv-badge" style="background:{color}">{row['category']}</span>
                    </div>
                    <div style="opacity:0.7; font-size:0.85rem; margin:0.2rem 0 0.4rem 0;">📅 {row['date']}</div>
                    <div style="font-size:0.9rem; opacity:0.9;">{meta.get('summary', '')}</div>
                    <div style="margin-top:0.4rem;">{skills_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3 — KNOWLEDGE GRAPH  (Module 3)
# ==============================================================================
with tab_graph:
    st.subheader("🕸 Knowledge Graph")
    st.caption("How your certifications, skills, projects, and internships connect — through shared skills.")

    docs_df = fetch_all_documents()
    skills_df = fetch_all_skills()
    rels_df = fetch_all_relationships()

    if docs_df.empty or skills_df.empty:
        st.info("Process some documents first — the graph will populate automatically as skills are linked.")
    else:
        if PYVIS_AVAILABLE:
            net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white", notebook=False)
            net.barnes_hut(gravity=-4000, spring_length=140)

            for _, row in docs_df.iterrows():
                meta = json.loads(row["metadata_json"] or "{}")
                color = CATEGORY_COLORS.get(row["category"], "#888")
                net.add_node(
                    f"doc_{row['id']}",
                    label=meta.get("title", row["filename"])[:28],
                    title=f"{row['category']} • {row['date']}\n{meta.get('summary','')}",
                    color=color, shape="dot", size=22,
                )

            for _, srow in skills_df.iterrows():
                net.add_node(
                    f"skill_{srow['id']}",
                    label=srow["name"],
                    title=f"Skill: {srow['name']}",
                    color="#ffffff", shape="diamond", size=14,
                )

            for _, rrow in rels_df.iterrows():
                src = f"{rrow['source_type']}_{rrow['source_id']}"
                tgt = f"{rrow['target_type']}_{rrow['target_id']}"
                net.add_edge(src, tgt, color="rgba(255,255,255,0.25)")

            net.set_options("""
            {
              "physics": {"stabilization": {"iterations": 120}},
              "interaction": {"hover": true, "navigationButtons": false}
            }
            """)

            html_path = os.path.join(STORAGE_DIR, "_graph.html")
            net.write_html(html_path, notebook=False)
            with open(html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=620, scrolling=True)
        else:
            st.warning("Install `pyvis` for the interactive network view (`pip install pyvis`). "
                       "Showing a connection matrix instead.")

        st.markdown("#### 📋 Document ↔ Skill Matrix")
        doc_titles = {}
        for _, row in docs_df.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            doc_titles[row["id"]] = f"{CATEGORY_ICONS.get(row['category'],'')} {meta.get('title', row['filename'])}"

        if not rels_df.empty:
            matrix_rows = []
            for _, row in docs_df.iterrows():
                doc_id = row["id"]
                linked_skill_ids = rels_df[
                    (rels_df["source_id"] == doc_id) & (rels_df["source_type"] == "document")
                ]["target_id"].tolist()
                linked_names = skills_df[skills_df["id"].isin(linked_skill_ids)]["name"].tolist()
                matrix_rows.append({
                    "Document": doc_titles[doc_id],
                    "Category": row["category"],
                    "Linked Skills": ", ".join(linked_names) if linked_names else "—",
                })
            st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

            st.markdown("#### 🔗 Cross-connections (docs sharing 2+ skills)")
            found_any = False
            doc_ids = docs_df["id"].tolist()
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    d1, d2 = doc_ids[i], doc_ids[j]
                    s1 = set(rels_df[(rels_df["source_id"] == d1) & (rels_df["source_type"] == "document")]["target_id"])
                    s2 = set(rels_df[(rels_df["source_id"] == d2) & (rels_df["source_type"] == "document")]["target_id"])
                    shared = s1 & s2
                    if len(shared) >= 1:
                        found_any = True
                        shared_names = skills_df[skills_df["id"].isin(shared)]["name"].tolist()
                        st.markdown(
                            f"- **{doc_titles[d1]}** ↔ **{doc_titles[d2]}** via "
                            + ", ".join(f"`{s}`" for s in shared_names)
                        )
            if not found_any:
                st.caption("No shared-skill connections yet — upload more related documents to build the chain.")


# ==============================================================================
# TAB 4 — SMART SEMANTIC SEARCH  (Module 5 — RAG)
# ==============================================================================
with tab_search:
    st.subheader("🔍 Smart Semantic Search")
    st.caption('Try: "Show all my AI projects", "certifications from 2024", "leadership achievements"')

    docs_df = fetch_all_documents()

    query = st.text_input("Search your portfolio", placeholder="e.g. Show all my AI projects")
    top_k = st.slider("Number of results", 1, 10, 5)
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

    if search_clicked and query.strip():
        if docs_df.empty:
            st.info("Nothing to search yet — upload some documents first.")
        else:
            # ------------------------------------------------------------------
            # RAG STEP 1: choose embedding backend (local hashing vs OpenAI API)
            # ------------------------------------------------------------------
            openai_client = None
            if use_openai_embeddings and api_key:
                openai_client = OpenAI(api_key=api_key)

            def embed(text):
                if openai_client is not None:
                    try:
                        return openai_embed(openai_client, text)
                    except Exception as e:
                        st.warning(f"OpenAI embedding call failed ({e}); falling back to local vectors.")
                        return hashing_embed(text)
                return hashing_embed(text)

            # ------------------------------------------------------------------
            # RAG STEP 2: embed the query
            # ------------------------------------------------------------------
            with st.spinner("Embedding query & scoring documents..."):
                query_vec = embed(query)

                # ------------------------------------------------------------------
                # RAG STEP 3: embed each stored document (its title+summary+skills+
                # text) and score with cosine similarity — this is the retrieval
                # half of Retrieval-Augmented Generation. (No generation step is
                # needed here since the "answer" IS the ranked set of source docs.)
                # ------------------------------------------------------------------
                scored = []
                for _, row in docs_df.iterrows():
                    corpus_text = build_document_corpus_text(row)
                    doc_vec = embed(corpus_text)
                    score = cosine_similarity(query_vec, doc_vec)
                    scored.append((score, row))

                scored.sort(key=lambda x: x[0], reverse=True)

            st.markdown(f"#### Top {min(top_k, len(scored))} results")
            for score, row in scored[:top_k]:
                meta = json.loads(row["metadata_json"] or "{}")
                color = CATEGORY_COLORS.get(row["category"], "#888")
                icon = CATEGORY_ICONS.get(row["category"], "📄")

                with st.container():
                    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
                    top_cols = st.columns([5, 2])
                    top_cols[0].markdown(f"**{icon} {meta.get('title', row['filename'])}**")
                    top_cols[1].metric("Relevance", f"{max(score, 0)*100:.0f}%")

                    badge_cols = st.columns([2, 2, 4])
                    badge_cols[0].markdown(
                        f'<span class="mv-badge" style="background:{color}">{row["category"]}</span>',
                        unsafe_allow_html=True,
                    )
                    badge_cols[1].caption(f"📅 {row['date']}")

                    st.write(meta.get("summary", ""))
                    if meta.get("extracted_skills"):
                        chips = "".join(f'<span class="mv-skill-chip">{s}</span>' for s in meta["extracted_skills"])
                        st.markdown(chips, unsafe_allow_html=True)

                    with st.expander("View extracted text"):
                        st.text(row["extracted_text"][:3000] if row["extracted_text"] else "—")

                    if os.path.exists(row["storage_path"]):
                        with open(row["storage_path"], "rb") as fh:
                            st.download_button(
                                "⬇️ Download original file", data=fh.read(),
                                file_name=row["filename"], key=f"dl_search_{row['id']}",
                            )
                    st.markdown('</div>', unsafe_allow_html=True)
    elif search_clicked:
        st.warning("Type something to search for first.")
