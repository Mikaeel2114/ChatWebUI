"""
Unicollect (پنل گفتگوی یونیکالکت): FastAPI app & OpenAI-compatible Router.

  /                       Unicollect Chat web panel (app/static)
  /v1/chat/completions    OpenAI-compatible Router endpoint (like 9router / OpenRouter)
  /v1/models              OpenAI-compatible models list
  /api/v1/*               Full public JSON & SSE API
  /api/v1/key             Returns backend-generated Public API Key & Route Addresses
  /api/chat               Chat endpoint (supports {message, history} and {prompt})
  /api/chat/stream        SSE token-by-token streaming endpoint
  /health                 Health check (used by the GitHub Pages status page + watchdog)
  /docs  /redoc           Interactive OpenAPI docs
  /api-docs               Persian API guide (same as GitHub Pages /api/)
  /public-key             Redirects to /site/public-key/ (Public API Key To Use page)
  /site/*                 The public site in pages/, served from the service too
  /chat                   Redirects to /site/chat/
"""
import hashlib
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

mimetypes.add_type("font/woff2", ".woff2")

VERSION = "2.3.0"
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PAGES_DIR = ROOT_DIR / "pages"
STATIC_DIR = BASE_DIR / "static"
KEY_FILE = ROOT_DIR / ".public_api_key"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("MODEL_NAME", os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_MATH_MODEL = os.environ.get("MATH_MODEL", "qwen3-vl:8b").strip() or "qwen3-vl:8b"
OLLAMA_ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemma3:1b").strip()
MODEL_LABEL = os.environ.get("MODEL_LABEL", "Unicollect Chat Mini")
MATH_MODEL_LABEL = os.environ.get("MATH_MODEL_LABEL", "Unicollect Math Model")
MODEL_ID = "unicollect-chat-mini"
MATH_MODEL_ID = "unicollect-math"


def _load_or_generate_api_key() -> str:
    env_key = os.environ.get("PUBLIC_API_KEY", "").strip()
    if env_key:
        try:
            KEY_FILE.write_text(env_key, encoding="utf-8")
        except OSError:
            pass
        return env_key
    if KEY_FILE.is_file():
        try:
            saved = KEY_FILE.read_text(encoding="utf-8").strip()
            if saved:
                return saved
        except OSError:
            pass
    generated = "sk-uc-" + secrets.token_hex(16)
    try:
        KEY_FILE.write_text(generated, encoding="utf-8")
    except OSError:
        pass
    return generated


PUBLIC_API_KEY = _load_or_generate_api_key()
APP_TOKEN = os.environ.get("APP_TOKEN", os.environ.get("PANEL_TOKEN", "")).strip()
DEMO_MODE = os.environ.get("PANEL_DEMO_MODE", "").lower() in ("1", "true", "yes")
RATE_LIMIT_RPM = int(os.environ.get("PANEL_RATE_LIMIT", "120"))

MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "8000"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "16"))
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", "0"))  # 0 = unlimited

DEFAULT_SYSTEM_PROMPT = (
    "تو دستیار هوش مصنوعی «یونیکالکت چت مینی» (Unicollect Chat Mini) هستی که توسط «موویکسو» (Muvixo) ساخته شده است.\n\n"
    "## هویت و لحن پاسخ‌گویی\n"
    "- همیشه در پاسخ به کاربر با **اول‌شخص («من»)** صحبت کن. هرگز به کاربر نگو «تو یونیکالکت هستی».\n"
    "- **مهم**: در پاسخ به سؤالات معمولی، علمی یا ریاضی، هرگز اول پاسخ خودت را معرفی نکن؛ مستقیم و بدون مقدمه برو سر اصل جواب. فقط وقتی کاربر صریحاً پرسید اسمت چیست یا کیستی بگو: «من **یونیکالکت چت مینی** هستم، دستیار هوش مصنوعی ساخته‌ی **موویکسو (Muvixo)**.»\n"
    "- اگر کاربر پرسید «چکارا می‌تونی بکنی؟» یا «چیا بلدی؟»، مستقیم و با فهرست کوتاه و مشخص توضیح بده چه کارهایی انجام می‌دهی (پاسخ به سؤال‌ها، حل مسائل ریاضی با فرمول، نوشتن و ویرایش متن و ایمیل، ترجمه، خلاصه‌سازی، ایده‌پردازی و آموزش مفاهیم). هرگز پاسخ کلی، تکراری یا بی‌ربط نده.\n"
    "- اگر کاربر درباره سازنده یا مدل زیرساخت پرسید، فقط بگو توسط **موویکسو (Muvixo)** ساخته شده‌ام و نام هیچ شرکت یا مدل دیگری (مثل Google، Gemma، Qwen، OpenAI و غیره) را نیاور.\n\n"
    "## قانون اکید زبان و فارسی روان\n"
    "- دقیقاً به همان زبان کاربر پاسخ بده (فارسی یا فینگیلیش ← فقط فارسی روان و طبیعی؛ انگلیسی ← انگلیسی؛ اسپانیایی ← اسپانیایی).\n"
    "- هرگز واژه‌های جعلی، شکسته یا نادرست فارسی نساز و هرگز از حروف چینی استفاده نکن (مگر کاربر چینی نوشته باشد).\n"
    "- اگر پیام کاربر فقط عدد یا عبارت ریاضی بود (مثل `52*95`)، به فارسی پاسخ بده.\n\n"
    "## دقت در ریاضیات و منطق\n"
    "- عملگرها را دقیق اجرا کن (`+` جمع، `-` تفریق، `*` ضرب، `/` تقسیم) و در مسائل چندمرحله‌ای، راه‌حل را مرحله‌به‌مرحله بنویس.\n"
    "- اگر در پیام یادداشت `[محاسبه دقیق تأییدشده: ...]` وجود داشت، دقیقاً همان نتیجه را گزارش بده.\n"
    "- اگر از بلوک تفکر (`<think>`) استفاده می‌کنی، تفکر داخلی را کوتاه (حداکثر چند خط) و **فقط به همان زبان کاربر** بنویس (اگر کاربر فارسی نوشت، داخل `<think>` هم ۱۰۰٪ به فارسی روان فکر کن و هرگز به چینی یا انگلیسی فکر نکن)، سپس پاسخ نهایی را کامل و مرتب به همان زبان بنویس.\n\n"
    "## ممنوعیت قطعی کدنویسی (قانون غیرقابل تغییر)\n"
    "- به هیچ عنوان کد برنامه‌نویسی، اسکریپت، کوئری، HTML/CSS/JS/Python یا دستور ترمینال ننویس و هیچ کدی را دیباگ یا بازنویسی نکن.\n"
    "- اگر کاربر درخواست نوشتن یا اصلاح کد کرد، محترمانه رد کن و به زبان خود کاربر بگو: «متأسفم، من برای کدنویسی طراحی نشده‌ام و کد نمی‌نویسم؛ اما می‌توانم منطق یا مفهوم آن را به زبان ساده برایت توضیح بدهم.»\n\n"
    "## فرمت‌بندی غنی\n"
    "- برای فرمول‌های داخل خط از `$...$` و برای فرمول‌های مستقل (کسر، رادیکال، انتگرال، ماتریس) از `$$...$$` استفاده کن.\n"
    "- برای ساختاردهی پاسخ از تیترها، لیست‌ها، جدول‌های مارک‌داون، چک‌لیست (`- [x]`)، کادر نکته (`> [!NOTE]`، `> [!TIP]`، `> [!WARNING]`)، هایلایت (`==متن==`) و بلوک‌های کد/متن با سه بکتیک (```...```) استفاده کن."
)

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT

MATH_SYSTEM_PROMPT = (
    "تو «Unicollect Math Model» (مدل تخصصی ریاضی یونیکالکت، ساخته‌ی موویکسو) هستی.\n\n"
    "## قوانین حیاتی\n"
    "1. **زبان ۱۰۰٪ یکدست**: اگر کاربر به فارسی نوشت، پاسخ نهایی را **۱۰۰٪ به فارسی روان و طبیعی** بنویس. هرگز از حروف چینی یا کلمات شکسته استفاده نکن.\n"
    "2. **محاسبه دقیق و بدون حدس**: تمام مراحل جبری و عددی را با دقت کامل انجام بده و هرگز نتیجه‌ی نهایی را حدس نزن؛ استدلال را منظم و بدون تکرار پیش ببر و سپس پاسخ نهایی را کامل و شفاف بنویس.\n"
    "3. **فرمت‌بندی ریاضی**: از LaTeX استاندارد (`$...$` برای داخل خط و `$$...$$` برای فرمول‌های مستقل) استفاده کن و پاسخ نهایی را واضح و مشخص بنویس.\n"
    "4. **دقت ریاضی**: اگر در پیام یادداشت `[محاسبه دقیق تأییدشده: ...]` وجود داشت، دقیقاً همان عدد تأییدشده را به‌عنوان نتیجه‌ی نهایی گزارش بده.\n"
    "5. **ممنوعیت کدنویسی**: هرگز کد برنامه‌نویسی یا اسکریپت ننویس؛ مسائل را فقط با منطق ریاضی و فرمول LaTeX حل کن.\n"
    "6. هرگز نام مدل یا شرکت دیگری (مثل Google، Gemma، Qwen، Phi، OpenAI) را نیاور و بدون مقدمه مستقیم برو سر اصل حل مسئله."
)

state = {
    "ollama_ready": DEMO_MODE,
    "last_error": None,
    "started_at": time.time(),
}

stats: Dict[str, Any] = {
    "requests": 0,
    "chat_calls": 0,
    "stream_calls": 0,
    "errors": 0,
    "latency_total_ms": 0,
}

_rate_buckets: Dict[str, List[float]] = defaultdict(list)


@asynccontextmanager
def _prewarm_models():
    if DEMO_MODE:
        return
    for m in dict.fromkeys([x for x in (OLLAMA_MODEL, OLLAMA_ROUTER_MODEL) if x]):
        try:
            requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": m, "prompt": "", "keep_alive": -1, "stream": False},
                timeout=180,
            )
        except Exception:
            pass


async def _lifespan(_app: FastAPI):
    threading.Thread(target=_prewarm_models, daemon=True).start()
    pub_url = os.environ.get("PUBLIC_URL", "http://localhost:8000").rstrip("/")
    print("=" * 72, flush=True)
    print(f"[Unicollect] Model                  : {MODEL_LABEL} (creator: Muvixo)", flush=True)
    print(f"[Unicollect] Backend Generated KEY  : {PUBLIC_API_KEY}", flush=True)
    print(f"[Unicollect] OpenAI / 9router Route : {pub_url}/v1", flush=True)
    print(f"[Unicollect] Chat Completions Route : {pub_url}/v1/chat/completions", flush=True)
    print(f"[Unicollect] Native Chat Route      : {pub_url}/api/v1/chat", flush=True)
    print(f"[Unicollect] Public Key Page        : {pub_url}/site/public-key/", flush=True)
    print("=" * 72, flush=True)
    yield


def _api_error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "status": status}},
        status_code=status,
    )


def _extract_supplied_key(req: Request) -> str:
    auth = req.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return (
        req.query_params.get("token")
        or req.query_params.get("api_key")
        or req.headers.get("X-API-Key", "")
        or req.headers.get("X-Panel-Token", "")
        or bearer
    ).strip()


def is_authorized(req: Request) -> bool:
    """If APP_TOKEN / PANEL_TOKEN is set, accept APP_TOKEN or PUBLIC_API_KEY."""
    supplied = _extract_supplied_key(req)
    if not APP_TOKEN:
        return True
    return supplied in (APP_TOKEN, PUBLIC_API_KEY)


def _check_rate_limit(req: Request) -> Optional[JSONResponse]:
    if RATE_LIMIT_RPM <= 0:
        return None
    ip = req.client.host if req.client else "unknown"
    now = time.time()
    bucket = [t for t in _rate_buckets[ip] if now - t < 60.0]
    _rate_buckets[ip] = bucket
    if len(bucket) >= RATE_LIMIT_RPM:
        return _api_error("rate_limited", f"Rate limit exceeded ({RATE_LIMIT_RPM}/min)", 429)
    bucket.append(now)
    return None


def _check_ollama() -> bool:
    if DEMO_MODE:
        state["ollama_ready"] = True
        state["last_error"] = None
        return True
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=3)
        ok = r.status_code == 200
        state["ollama_ready"] = ok
        state["last_error"] = None if ok else f"HTTP {r.status_code}"
        return ok
    except requests.RequestException as exc:
        state["ollama_ready"] = False
        state["last_error"] = str(exc)
        return False


def _load_not_found() -> str:
    try:
        return (PAGES_DIR / "404.html").read_text(encoding="utf-8").replace("__PAGES_BASE__", "/site/")
    except OSError:
        return ""


def _asset_version() -> str:
    h = hashlib.sha1()
    for f in ("style.css", "script.js"):
        try:
            h.update((STATIC_DIR / f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:10]


def _load_index_html() -> str:
    try:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8").replace("__V__", _asset_version())
    except OSError:
        return "<h1>Unicollect · پنل گفتگوی یونیکالکت</h1>"


_NOT_FOUND_HTML = _load_not_found()
_INDEX_HTML = _load_index_html()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    history: List[ChatMessage] = []
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = None
    model: Optional[str] = None


class OpenAIChatIn(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.5
    max_tokens: Optional[int] = None


app = FastAPI(
    title="Unicollect Chat API & Router",
    version=VERSION,
    lifespan=_lifespan,
    description=(
        "یونیکالکت چت مینی — دستیار هوش مصنوعی موویکسو (Muvixo) با استریم زنده و روتر سازگار با OpenAI / 9router.\n\n"
        f"**Public API Key (KEY):** `{PUBLIC_API_KEY}`\n\n"
        "- **Base URL (OpenAI / 9router SDKs):** `/v1`\n"
        "- **Chat Completions Route:** `POST /v1/chat/completions` (or `/api/v1/chat/completions`)\n"
        "- **Models Route:** `GET /v1/models`\n"
        "- **Native Chat Route:** `POST /api/v1/chat`\n\n"
        "Persian guide: [/api-docs](/api-docs) · Public Key Page: [/public-key](/public-key)"
    ),
    openapi_tags=[
        {"name": "chat", "description": "Chat, streaming, and OpenAI-compatible router endpoints."},
        {"name": "discovery", "description": "Public API Key, route addresses, status, models, config, and stats."},
    ],
)

_origins = [o.strip() for o in os.environ.get("PANEL_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Panel-Token", "Authorization"],
)


import ast
import operator as _op

_SAFE_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
    ast.Pow: _op.pow, ast.USub: _op.neg, ast.UAdd: _op.pos,
}


def _eval_math_hint(text: str) -> Optional[str]:
    """Deterministically evaluate pure arithmetic expressions ONLY (never inside LaTeX or algebra)."""
    raw = (text or "").strip()
    if not raw or any(ch in raw for ch in ("\\", "$", "^", "_", "{", "}")):
        return None
    if re.search(r"[A-Za-z]", raw):
        return None
    norm = raw.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩×÷", "01234567890123456789*/"))
    # Strip common Persian arithmetic question wrappers only
    stripped = re.sub(r"(?:چند\s*م[ی‌]?(?:شود|شه)|حساب\s*کن|محاسبه\s*کن|برابر\s*است\s*با|چنده|مقدار|[؟?=])", " ", norm).strip()
    if not re.fullmatch(r"[0-9\s+\-*/().%]+", stripped) or not re.search(r"[+\-*/%]", stripped):
        return None
    cleaned = stripped
    if len(cleaned) > 80:
        return None
    try:
        node = ast.parse(cleaned, mode="eval").body
        def _ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
                return _SAFE_OPS[type(n.op)](_ev(n.operand))
            if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
                return _SAFE_OPS[type(n.op)](_ev(n.left), _ev(n.right))
            raise ValueError("unsupported")
        val = _ev(node)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        elif isinstance(val, float):
            val = round(val, 6)
        return f"{cleaned} = {val}"
    except Exception:
        return None


def _needs_persian_anchor(text: str) -> bool:
    has_persian = bool(re.search(r"[\u0600-\u06FF]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    return not has_persian and not has_latin


_INSTALLED_MODELS_CACHE: Dict[str, Any] = {"ts": 0.0, "names": set()}

_MATH_KEYWORDS_RE = re.compile(
    r"(?:\\(?:frac|dfrac|tfrac|sqrt|int|iint|sum|prod|lim|sin|cos|tan|cot|log|ln|in|notin|subset|cup|cap|mathbb|vec|hat|bar|overline|begin\{[pbv]?matrix\})|"
    r"\$\$|\$[^$\n]+\$|\\\(|\\\[|"
    r"[∈∉⊂⊆∪∩∅∀∃∑∏∫∬∮√∞±∓×÷≤≥≠≈≡∝ℝℕℤℚℂ°∠⊥∥⇒⇔→←]|"
    r"معادله|نامعادله|انتگرال|مشتق|دیفرانسیل|ماتریس|دترمینان|رادیکال|لگاریتم|مثلثات|فیثاغورث|چندجمله‌ای|"
    r"مجموعه|زیرمجموعه|عضو\s|اعداد\s+(?:صحیح|طبیعی|حقیقی|گویا|گنگ|مختلط|اول)|"
    r"تابع|دامنه|برد\s|مقدار|حاصل|جذر|کسر|توان|ضرب|تقسیم|تفریق|مقسوم|دنباله|بردار|مختصات|شیب\s|معکوس|"
    r"احتمال|آمار|هندسه|جبر|ریاضی|مساحت|محیط|حجم|زاویه|مثلث|دایره|مربع|مستطیل|استوانه|مخروط|کره|حد\s|"
    r"\b(?:solve|integral|derivative|matrix|determinant|equation|algebra|calculus|theorem|polynomial|eigenvalue|quadratic|factorial|logarithm|subset|function|domain|range|probability|geometry|trigonometry)\b)",
    re.IGNORECASE,
)


def _is_math_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _eval_math_hint(t) is not None:
        return True
    if _MATH_KEYWORDS_RE.search(t):
        return True
    # Algebraic expressions like 2x + 5 = 15, x^2 - 4 = 0, or f(x) = ...
    if re.search(r"(?:\d\s*[a-zA-Z]|[a-zA-Z]\s*\^\s*\d|[fgh]\s*\(\s*[a-zA-Z]\s*\)).*=", t):
        return True
    # Persian/Arabic digits with arithmetic operators
    norm = t.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩×÷", "01234567890123456789*/"))
    if re.search(r"\d+\s*[+\-*/^=]\s*\d+", norm):
        return True
    return False


def _has_ollama_model(model_name: str) -> bool:
    if not model_name or DEMO_MODE:
        return False
    now = time.time()
    if not _INSTALLED_MODELS_CACHE["names"] or (now - _INSTALLED_MODELS_CACHE["ts"] > 30):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4)
            if r.ok:
                names = {str(m.get("name", "")).strip().lower() for m in r.json().get("models", []) if m.get("name")}
                if names:
                    _INSTALLED_MODELS_CACHE["names"] = names
                    _INSTALLED_MODELS_CACHE["ts"] = now
        except Exception:
            return False
    target = model_name.strip().lower()
    names = _INSTALLED_MODELS_CACHE["names"]
    if target in names or f"{target}:latest" in names:
        return True
    # Match if user passed base name without tag
    if ":" not in target and any(n.startswith(target + ":") for n in names):
        return True
    return False


_ROUTER_SYSTEM_PROMPT = (
    "You are a strict intent router. Output MATH ONLY if the user is asking you to solve, calculate, prove, or explain an actual mathematical problem or equation.\n"
    "Output CHAT for EVERYTHING ELSE — including greetings, writing, translation, system prompts, rules, guidelines, or messages that merely mention the words math/ریاضی/LaTeX without giving a math problem to solve.\n"
    "Examples:\n"
    "User: سلام -> CHAT\n"
    "User: ساده -> CHAT\n"
    "User: تو یونیکالکت هستی، فقط وقتی فرمول ریاضی وجود دارد از LaTeX استفاده کن -> CHAT\n"
    "User: مجموعه فیلم خوب معرفی کن -> CHAT\n"
    "User: 52 * 95 چند میشه؟ -> MATH\n"
    "User: اگر a ∈ ℤ و a ∈ A باشد -> MATH\n"
    "User: ثابت کنید حاصلضرب زیر یک عدد گویاست -> MATH\n"
    "Reply with ONLY one word: CHAT or MATH."
)


def _has_any_math_signal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"[0-9۰-۹٠-٩=+\-*/^%$∈∉⊂⊆∪∩∅∀∃∑∏∫√∞±×÷≤≥≠≈ℝℕℤℚℂ\\]", t):
        return True
    return _is_math_query(t)


_DEFINITE_MATH_RE = re.compile(
    r"(?:\\(?:frac|dfrac|tfrac|sqrt|prod|sum|int|iint|lim|binom|mathbb|in|notin|subseteq|begin\{)|"
    r"\$\$|\$[^$\n]+\$|\\\(|\\\[|"
    r"[∈∉⊂⊆∪∩∅∀∃∑∏∫∬∮√∞±∓×÷≤≥≠≈≡∝ℝℕℤℚℂ])"
)


def _classify_with_router_model(user_text: str) -> Optional[bool]:
    """Three-tier hybrid router:
    1) Definite math (arithmetic like ۵+۶ or LaTeX/Unicode math) -> True (0ms)
    2) Definite chat (no digits, symbols, or math terms like ساده) -> False (0ms)
    3) Ambiguous cases -> lightweight router model (gemma3:1b) decides CHAT vs MATH."""
    snippet = (user_text or "").strip()[:1000]
    if not snippet:
        return False
    if _eval_math_hint(snippet) is not None or _DEFINITE_MATH_RE.search(snippet):
        return True
    if not _has_any_math_signal(snippet):
        return False
    if not OLLAMA_ROUTER_MODEL or DEMO_MODE or not _has_ollama_model(OLLAMA_ROUTER_MODEL):
        return None
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": snippet},
                ],
                "temperature": 0.0,
                "max_tokens": 6,
                "keep_alive": -1,
                "stream": False,
            },
            timeout=8,
        )
        if not r.ok:
            return None
        r.encoding = "utf-8"
        ans = (r.json()["choices"][0]["message"].get("content") or "").strip().upper()
        if "MATH" in ans and "CHAT" not in ans:
            return True
        if "CHAT" in ans and "MATH" not in ans:
            return False
    except Exception:
        return None
    return None


def _select_backend_model(user_text: str, requested_model: Optional[str] = None):
    req = (requested_model or "").strip().lower()
    if req in (MATH_MODEL_ID, "math", "unicollect-math", OLLAMA_MATH_MODEL.lower()):
        want_math = True
    elif req in (MODEL_ID, "chat", "unicollect-chat-mini", OLLAMA_MODEL.lower()):
        want_math = False
    else:
        routed = _classify_with_router_model(user_text)
        want_math = routed if routed is not None else _is_math_query(user_text)
    if want_math:
        if OLLAMA_MATH_MODEL and _has_ollama_model(OLLAMA_MATH_MODEL):
            return OLLAMA_MATH_MODEL, MATH_MODEL_LABEL, True
        return OLLAMA_MODEL, MATH_MODEL_LABEL, True
    return OLLAMA_MODEL, MODEL_LABEL, False


def _validate_and_build_messages(data: ChatIn):
    text = (data.message if data.message is not None else data.prompt or "").strip()
    if not text:
        return None, _api_error("empty_message", "پیام خالی است", 400)
    if len(text) > MAX_MESSAGE_LENGTH:
        return None, _api_error(
            "message_too_long",
            f"پیام خیلی طولانی است (حداکثر {MAX_MESSAGE_LENGTH} نویسه)",
            400,
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in data.history[-(MAX_HISTORY_TURNS * 2):]:
        if m.role in ("user", "assistant") and m.content.strip():
            messages.append({"role": m.role, "content": m.content.strip()[:MAX_MESSAGE_LENGTH]})

    math_hint = _eval_math_hint(text)
    if math_hint and _needs_persian_anchor(text):
        user_content = f"{text}\n[محاسبه دقیق تأییدشده: {math_hint}]\n(لطفاً فقط به زبان فارسی پاسخ بده و دقیقاً همین نتیجه را بگو.)"
    elif math_hint:
        user_content = f"{text}\n[Verified exact calculation: {math_hint}. Reply in the exact same language as the user.]"
    elif _needs_persian_anchor(text):
        user_content = f"{text}\n(لطفاً فقط به زبان فارسی پاسخ بده و هرگز به زبان چینی ننویس.)"
    else:
        user_content = text
    messages.append({"role": "user", "content": user_content})
    return messages, None


def _route_bundle(req: Request) -> Dict[str, Any]:
    base = str(req.base_url).rstrip("/")
    return {
        "api_key": PUBLIC_API_KEY,
        "public_api_key": PUBLIC_API_KEY,
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "creator": "Muvixo",
        "base_url": base,
        "routes": {
            "openai_base_url": f"{base}/v1",
            "chat_completions": f"{base}/v1/chat/completions",
            "v1_chat_completions": f"{base}/api/v1/chat/completions",
            "models": f"{base}/v1/models",
            "native_chat": f"{base}/api/v1/chat",
            "native_stream": f"{base}/api/v1/chat/stream",
            "health": f"{base}/api/v1/health",
            "public_key_page": f"{base}/site/public-key/",
        },
    }


@app.get("/api/v1/key", tags=["discovery"])
@app.get("/v1/key", tags=["discovery"])
def get_public_key(request: Request):
    """Returns the backend-generated Public API Key (KEY) and all router addresses."""
    return _route_bundle(request)


@app.get("/health", tags=["discovery"])
@app.get("/api/v1/health", tags=["discovery"])
def health(request: Request):
    _check_ollama()
    rb = _route_bundle(request)
    return {
        "status": "ok" if state["ollama_ready"] else "loading",
        "mode": "demo" if DEMO_MODE else "ollama",
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "ollama_model": MODEL_LABEL,
        "creator": "Muvixo",
        "public_api_key": PUBLIC_API_KEY,
        "api_key": PUBLIC_API_KEY,
        "routes": rb["routes"],
        "auth_required": bool(APP_TOKEN),
        "uptime_sec": round(time.time() - state["started_at"], 1),
        "error": state["last_error"],
    }


@app.get("/api/v1/ready", tags=["discovery"])
def ready():
    if _check_ollama():
        return {"ready": True, "model": MODEL_LABEL, "public_api_key": PUBLIC_API_KEY}
    return JSONResponse({"ready": False, "error": state["last_error"]}, status_code=503)


@app.get("/v1", tags=["discovery"])
@app.get("/api/v1", tags=["discovery"])
@app.get("/api/info", tags=["discovery"])
def api_info(request: Request):
    rb = _route_bundle(request)
    return {
        "service": "ollama-web-panel",
        "name": "یونیکالکت چت مینی",
        "creator": "Muvixo",
        "version": VERSION,
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "public_api_key": PUBLIC_API_KEY,
        "api_key": PUBLIC_API_KEY,
        "routes": rb["routes"],
        "auth_required": bool(APP_TOKEN),
        "endpoints": {
            "key": "/api/v1/key",
            "openai_base": "/v1",
            "openai_models": "/v1/models",
            "openai_completions": "/v1/chat/completions",
            "v1_completions": "/api/v1/chat/completions",
            "health": "/api/v1/health",
            "ready": "/api/v1/ready",
            "info": "/api/v1/info",
            "models": "/api/v1/models",
            "config": "/api/v1/config",
            "stats": "/api/v1/stats",
            "chat": "/api/v1/chat",
            "chat_stream": "/api/v1/chat/stream",
            "docs": "/docs",
            "redoc": "/redoc",
            "guide": "/api-docs",
            "public_key_page": "/public-key",
            "site": "/site/",
        },
    }


@app.get("/api/v1/info", tags=["discovery"])
def v1_info(request: Request):
    rb = _route_bundle(request)
    return {
        "service": "ollama-web-panel",
        "name": "یونیکالکت چت مینی",
        "creator": "Muvixo",
        "version": VERSION,
        "mode": "demo" if DEMO_MODE else "ollama",
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "public_api_key": PUBLIC_API_KEY,
        "routes": rb["routes"],
        "limits": {
            "max_message_length": MAX_MESSAGE_LENGTH,
            "max_history_turns": MAX_HISTORY_TURNS,
            "max_reply_tokens": MAX_REPLY_TOKENS,
            "rate_limit_rpm": RATE_LIMIT_RPM,
        },
    }


@app.get("/api/v1/config", tags=["discovery"])
def v1_config(request: Request):
    rb = _route_bundle(request)
    return {
        "auth_required": bool(APP_TOKEN),
        "public_api_key": PUBLIC_API_KEY,
        "api_key": PUBLIC_API_KEY,
        "routes": rb["routes"],
        "rate_limit_rpm": RATE_LIMIT_RPM,
        "max_message_length": MAX_MESSAGE_LENGTH,
        "max_history_turns": MAX_HISTORY_TURNS,
        "max_reply_tokens": MAX_REPLY_TOKENS,
        "default_model": MODEL_LABEL,
        "model_id": MODEL_ID,
    }


@app.get("/v1/models", tags=["discovery"])
@app.get("/api/v1/models", tags=["discovery"])
def v1_models():
    now_ts = int(time.time())
    return {
        "object": "list",
        "default": MODEL_LABEL,
        "public_api_key": PUBLIC_API_KEY,
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": now_ts,
                "owned_by": "muvixo",
                "label": MODEL_LABEL,
            },
            {
                "id": MATH_MODEL_ID,
                "object": "model",
                "created": now_ts,
                "owned_by": "muvixo",
                "label": MATH_MODEL_LABEL,
            },
            {
                "id": MODEL_LABEL,
                "object": "model",
                "created": now_ts,
                "owned_by": "muvixo",
                "label": MODEL_LABEL,
            },
        ],
        "models": [
            {
                "id": MODEL_ID,
                "name": MODEL_LABEL,
                "label": MODEL_LABEL,
                "creator": "Muvixo",
                "active": True,
            },
            {
                "id": MATH_MODEL_ID,
                "name": MATH_MODEL_LABEL,
                "label": MATH_MODEL_LABEL,
                "creator": "Muvixo",
                "active": True,
            },
        ],
    }


@app.get("/api/v1/stats", tags=["discovery"])
def v1_stats():
    avg_ms = round(stats["latency_total_ms"] / stats["chat_calls"]) if stats["chat_calls"] else 0
    return {
        **stats,
        "avg_latency_ms": avg_ms,
        "uptime_sec": round(time.time() - state["started_at"], 1),
    }


@app.post("/api/chat", tags=["chat"])
@app.post("/api/v1/chat", tags=["chat"])
@app.post("/api/v1/generate", tags=["chat"])
def chat(data: ChatIn, request: Request):
    stats["requests"] += 1
    if not is_authorized(request):
        return _api_error("unauthorized", "Unauthorized", 401)
    if rl := _check_rate_limit(request):
        return rl

    messages, err = _validate_and_build_messages(data)
    if err:
        stats["errors"] += 1
        return err

    started = time.time()
    stats["chat_calls"] += 1
    if DEMO_MODE:
        raw_user = (data.message if data.message is not None else data.prompt or "").strip()
        reply = f"[Demo Mode · {MODEL_LABEL}] {raw_user}"
        ms = round((time.time() - started) * 1000)
        stats["latency_total_ms"] += ms
        return {"reply": reply, "response": reply, "model": MODEL_LABEL, "latency_ms": ms}

    raw_user = (data.message if data.message is not None else data.prompt or "").strip()
    backend_model, active_label, is_math = _select_backend_model(raw_user, getattr(data, "model", None))
    chat_temp = 0.25 if is_math else (data.temperature if data.temperature is not None else 0.5)
    run_messages = list(messages)
    if is_math:
        # Keep only the system prompt + current question (unless user explicitly refers to previous turn) to prevent topic contamination
        is_followup = bool(re.search(r"(?:قبلی|ادامه|همین|بالا|مرحله|توضیح بیشتر|چرا این|again|above|previous|continue)", raw_user, re.I))
        if not is_followup and len(run_messages) > 2:
            run_messages = [run_messages[0], run_messages[-1]]
        elif len(run_messages) > 5:
            run_messages = [run_messages[0]] + run_messages[-4:]
        if run_messages and run_messages[0].get("role") == "system":
            run_messages[0] = {"role": "system", "content": MATH_SYSTEM_PROMPT}
        else:
            run_messages.insert(0, {"role": "system", "content": MATH_SYSTEM_PROMPT})
        if re.search(r"[\u0600-\u06FF]", raw_user) and run_messages and run_messages[-1].get("role") == "user":
            run_messages[-1] = {"role": "user", "content": run_messages[-1]["content"] + "\n(فقط به همین سؤال پاسخ بده؛ استدلال را دقیق و بدون تکرار بنویس و پاسخ نهایی را کامل و ۱۰۰٪ به فارسی روان با فرمول LaTeX ارائه کن.)"}
    chat_payload = {
        "model": backend_model,
        "messages": run_messages,
        "temperature": chat_temp,
        "top_p": 0.9,
        "keep_alive": -1,
        "frequency_penalty": 0.35,
        "presence_penalty": 0.2,
    }
    chat_payload["max_tokens"] = data.max_tokens or (MAX_REPLY_TOKENS if MAX_REPLY_TOKENS > 0 else 8192)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json=chat_payload,
            timeout=600,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        payload = json.loads(resp.content.decode("utf-8"))
        reply = payload["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        stats["errors"] += 1
        return _api_error("upstream_unavailable", f"سرویس مدل در دسترس نیست: {exc}", 502)
    except (KeyError, IndexError, ValueError):
        stats["errors"] += 1
        return _api_error("invalid_upstream_response", "پاسخ نامعتبر از سرویس مدل", 502)

    ms = round((time.time() - started) * 1000)
    stats["latency_total_ms"] += ms
    return {"reply": reply, "response": reply, "model": active_label, "latency_ms": ms}



class TitleIn(BaseModel):
    message: str
    reply: Optional[str] = ""


@app.post("/api/chat/title", tags=["chat"])
@app.post("/api/v1/title", tags=["chat"])
def chat_title(data: TitleIn, request: Request):
    if not is_authorized(request):
        return _api_error("unauthorized", "Unauthorized", 401)
    msg = (data.message or "").strip()[:400]
    if not msg:
        return {"title": "گفتگوی جدید"}
    if DEMO_MODE:
        return {"title": msg[:36]}
    title_model = OLLAMA_ROUTER_MODEL if (OLLAMA_ROUTER_MODEL and _has_ollama_model(OLLAMA_ROUTER_MODEL)) else OLLAMA_MODEL
    has_fa = bool(re.search(r"[\u0600-\u06FF]", msg))
    sys_p = (
        "یک عنوان خیلی کوتاه (حداکثر ۳ تا ۵ کلمه) برای این گفتگو بنویس. فقط خود عنوان را بدون علامت نقل‌قول، بدون نقطه و بدون LaTeX برگردان."
        if has_fa
        else "Write a very short 3 to 5 word title for this chat. Return ONLY the plain title with no quotes, no period, and no LaTeX."
    )
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": title_model,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": msg},
                ],
                "temperature": 0.2,
                "max_tokens": 24,
                "stream": False,
            },
            timeout=15,
        )
        if r.ok:
            r.encoding = "utf-8"
            raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
            clean = re.sub(r'["\x27«»`*_#:\n\r]+', " ", raw).strip()
            clean = re.sub(r"\s+", " ", clean)
            if clean:
                return {"title": clean[:45]}
    except Exception:
        pass
    return {"title": ""}


@app.post("/api/chat/stream", tags=["chat"])
@app.post("/api/v1/chat/stream", tags=["chat"])
def chat_stream(data: ChatIn, request: Request):
    stats["requests"] += 1
    if not is_authorized(request):
        return _api_error("unauthorized", "Unauthorized", 401)
    if rl := _check_rate_limit(request):
        return rl

    messages, err = _validate_and_build_messages(data)
    if err:
        stats["errors"] += 1
        return err

    stats["stream_calls"] += 1

    def sse(obj) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def gen():
        if DEMO_MODE:
            yield sse({"meta": {"model": MODEL_LABEL, "is_math": False}})
            demo_text = f"[Demo Mode · {MODEL_LABEL}] سلام! سرویس در حالت آزمایشی آماده است."
            for token in demo_text.split(" "):
                yield sse({"t": token + " "})
            yield "data: [DONE]\n\n"
            return

        raw_user = (data.message if data.message is not None else data.prompt or "").strip()
        yield sse({"phase": "routing"})
        route_t0 = time.time()
        backend_model, active_label, is_math = _select_backend_model(raw_user, getattr(data, "model", None))
        route_ms = round((time.time() - route_t0) * 1000)
        yield sse({"meta": {"model": active_label, "is_math": is_math, "route_ms": route_ms}})
        stream_temp = 0.25 if is_math else (data.temperature if data.temperature is not None else 0.5)
        run_messages = list(messages)
        if is_math:
            is_followup = bool(re.search(r"(?:قبلی|ادامه|همین|بالا|مرحله|توضیح بیشتر|چرا این|again|above|previous|continue)", raw_user, re.I))
            if not is_followup and len(run_messages) > 2:
                run_messages = [run_messages[0], run_messages[-1]]
            elif len(run_messages) > 5:
                run_messages = [run_messages[0]] + run_messages[-4:]
            if run_messages and run_messages[0].get("role") == "system":
                run_messages[0] = {"role": "system", "content": MATH_SYSTEM_PROMPT}
            else:
                run_messages.insert(0, {"role": "system", "content": MATH_SYSTEM_PROMPT})
            if re.search(r"[\u0600-\u06FF]", raw_user) and run_messages and run_messages[-1].get("role") == "user":
                run_messages[-1] = {"role": "user", "content": run_messages[-1]["content"] + "\n(فقط به همین سؤال پاسخ بده؛ استدلال را دقیق و بدون تکرار بنویس و پاسخ نهایی را کامل و ۱۰۰٪ به فارسی روان با فرمول LaTeX ارائه کن.)"}
        stream_payload = {
            "model": backend_model,
            "messages": run_messages,
            "temperature": stream_temp,
            "top_p": 0.9,
            "keep_alive": -1,
            "frequency_penalty": 0.35,
            "presence_penalty": 0.2,
            "stream": True,
        }
        stream_payload["max_tokens"] = data.max_tokens or (MAX_REPLY_TOKENS if MAX_REPLY_TOKENS > 0 else 8192)
        try:
            upstream = requests.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json=stream_payload,
                timeout=600,
                stream=True,
            )
        except requests.RequestException as exc:
            yield sse({"error": f"سرویس مدل در دسترس نیست: {exc}"})
            return

        if upstream.status_code != 200:
            yield sse({"error": f"سرویس مدل خطا داد (HTTP {upstream.status_code})"})
            return

        upstream.encoding = "utf-8"
        got_any = False
        acc_answer = ""
        try:
            for raw_bytes in upstream.iter_lines():
                if not raw_bytes:
                    continue
                line = raw_bytes.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    d_obj = obj["choices"][0]["delta"]
                    delta = d_obj.get("content", "") or ""
                    r_delta = d_obj.get("reasoning", "") or d_obj.get("reasoning_content", "") or ""
                except (ValueError, KeyError, IndexError, AttributeError):
                    delta = ""
                    r_delta = ""
                if r_delta:
                    got_any = True
                    yield sse({"r": r_delta})
                if delta:
                    got_any = True
                    acc_answer += delta
                    # Break immediately if the model enters a verbatim repetition loop (120+ chars repeated)
                    if len(acc_answer) > 360:
                        tail = acc_answer[-120:]
                        if acc_answer[:-120].rfind(tail) != -1:
                            break
                    yield sse({"t": delta})
        except requests.RequestException as exc:
            if not got_any:
                yield sse({"error": f"ارتباط با سرویس مدل قطع شد: {exc}"})
                return
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/chat/completions", tags=["chat"])
@app.post("/api/v1/chat/completions", tags=["chat"])
def openai_chat_completions(data: OpenAIChatIn, request: Request):
    """OpenAI-compatible /v1/chat/completions router endpoint (works like 9router / OpenRouter)."""
    stats["requests"] += 1
    if not is_authorized(request):
        return _api_error("unauthorized", "Unauthorized", 401)
    if rl := _check_rate_limit(request):
        return rl
    if not data.messages:
        return _api_error("empty_messages", "messages cannot be empty", 400)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in data.messages:
        if m.role in ("user", "assistant") and m.content.strip():
            txt = m.content.strip()[:MAX_MESSAGE_LENGTH]
            if m.role == "user" and _needs_persian_anchor(txt):
                txt = f"{txt}\n(لطفاً فقط به زبان فارسی پاسخ بده و هرگز به زبان چینی ننویس.)"
            msgs.append({"role": m.role, "content": txt})

    if DEMO_MODE:
        last_user = next((m.content for m in reversed(data.messages) if m.role == "user"), "")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_LABEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"[Demo Mode · {MODEL_LABEL}] {last_user}"},
                "finish_reason": "stop",
            }],
        }

    last_u = next((m.content for m in reversed(data.messages) if m.role == "user"), "")
    backend_model, oai_label, is_math = _select_backend_model(last_u, data.model)
    oai_payload = {
        "model": backend_model,
        "messages": msgs,
        "temperature": 0.25 if is_math else data.temperature,
        "stream": data.stream,
        "keep_alive": -1,
    }
    oai_payload["max_tokens"] = data.max_tokens or (MAX_REPLY_TOKENS if MAX_REPLY_TOKENS > 0 else 8192)
    try:
        upstream = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json=oai_payload,
            timeout=600,
            stream=data.stream,
        )
        upstream.raise_for_status()
        if data.stream:
            def relay():
                for chunk in upstream.iter_lines():
                    if chunk:
                        yield chunk.decode("utf-8", errors="replace") + "\n\n"
            return StreamingResponse(relay(), media_type="text/event-stream; charset=utf-8")
        upstream.encoding = "utf-8"
        res_json = json.loads(upstream.content.decode("utf-8"))
        res_json["model"] = oai_label
        return JSONResponse(res_json)
    except requests.RequestException as exc:
        stats["errors"] += 1
        return _api_error("upstream_unavailable", str(exc), 502)


@app.get("/")
def index(request: Request):
    if not is_authorized(request):
        return HTMLResponse(
            "Unauthorized. Append <code>?token=YOUR_TOKEN</code> to the URL.",
            status_code=401,
        )
    return HTMLResponse(_INDEX_HTML, headers={"Cache-Control": "no-cache"})



@app.get("/chat")
def chat_redirect():
    return RedirectResponse("/site/chat/")



@app.get("/api-docs")
def api_docs_redirect():
    return RedirectResponse("/site/api/")


@app.get("/public-key")
def public_key_redirect():
    return RedirectResponse("/site/public-key/")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if PAGES_DIR.is_dir():
    app.mount("/site", StaticFiles(directory=str(PAGES_DIR), html=True), name="site")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and "text/html" in request.headers.get("accept", "") and _NOT_FOUND_HTML:
        return HTMLResponse(_NOT_FOUND_HTML, status_code=404)
    return _api_error("http_error", str(exc.detail), exc.status_code)
