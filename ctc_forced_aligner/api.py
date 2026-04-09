import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .align import TORCH_DTYPES
from .alignment_utils import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
)
from .text_utils import postprocess_results, preprocess_text

LANGUAGE_TO_ISO = {
    "en": "eng",
    "zh": "chi",
}

def tokenize_by_CJK_char(line: str) -> str:
    """
    Tokenize a line of text with CJK char.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 hello world 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip())
    return " ".join([w.strip() for w in chars if w.strip()])

def _resolve_language_code(language: str) -> str:
    language = language.strip().lower()
    if language in LANGUAGE_TO_ISO:
        return LANGUAGE_TO_ISO[language]
    if len(language) == 3 and language.isalpha():
        return language
    raise ValueError("language must be one of: en, zh, or a valid ISO-639-3 code")


def _env_int(name: str, default_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default_value
    try:
        return int(raw)
    except ValueError:
        return default_value


def _env_float(name: str, default_value: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default_value
    try:
        return float(raw)
    except ValueError:
        return default_value


def _build_runtime_config() -> dict:
    has_cuda = torch.cuda.is_available()
    default_device = "cuda" if has_cuda else "cpu"
    device = os.getenv("ALIGN_DEVICE", default_device)

    default_dtype = "float16" if device == "cuda" else "float32"
    compute_dtype = os.getenv("ALIGN_COMPUTE_DTYPE", default_dtype)
    if compute_dtype not in TORCH_DTYPES:
        compute_dtype = default_dtype

    default_concurrency = 1 if device == "cuda" else max(1, min(4, os.cpu_count() or 1))
    max_concurrency = max(1, _env_int("ALIGN_MAX_CONCURRENCY", default_concurrency))

    return {
        "alignment_model": os.getenv(
            "ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner"
        ),
        "device": device,
        "compute_dtype": compute_dtype,
        "attn_implementation": os.getenv("ALIGN_ATTN_IMPLEMENTATION", "") or None,
        "window_size": _env_float("ALIGN_WINDOW_SIZE", 30.0),
        "context_size": _env_float("ALIGN_CONTEXT_SIZE", 2.0),
        "batch_size": max(1, _env_int("ALIGN_BATCH_SIZE", 4)),
        "max_concurrency": max_concurrency,
        "max_workers": max(1, _env_int("ALIGN_MAX_WORKERS", max_concurrency)),
    }


RUNTIME_CONFIG = _build_runtime_config()


class AlignmentResponse(BaseModel):
    text: str
    language: str
    segments: list[dict]
    model: str
    device: str


app = FastAPI(title="CTC Forced Aligner API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    model, tokenizer = load_alignment_model(
        RUNTIME_CONFIG["device"],
        RUNTIME_CONFIG["alignment_model"],
        RUNTIME_CONFIG["attn_implementation"],
        TORCH_DTYPES[RUNTIME_CONFIG["compute_dtype"]],
    )
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.semaphore = asyncio.Semaphore(RUNTIME_CONFIG["max_concurrency"])
    app.state.executor = ThreadPoolExecutor(max_workers=RUNTIME_CONFIG["max_workers"])


@app.on_event("shutdown")
def _shutdown() -> None:
    executor: Optional[ThreadPoolExecutor] = getattr(app.state, "executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": RUNTIME_CONFIG["alignment_model"],
        "device": RUNTIME_CONFIG["device"],
        "max_concurrency": RUNTIME_CONFIG["max_concurrency"],
    }



def _run_alignment(audio_path: str, text: str, language_code: str, romanize: bool, split_size: str,
                   star_frequency: str, merge_threshold: float, batch_size: int, window_size: float,
                   context_size: float) -> list[dict]:
    model = app.state.model
    tokenizer = app.state.tokenizer

    audio_waveform = load_audio(audio_path, model.dtype, model.device)
    emissions, stride = generate_emissions(
        model,
        audio_waveform,
        window_length=window_size,
        context_length=context_size,
        batch_size=batch_size,
    )

    text = tokenize_by_CJK_char(text)

    tokens_starred, text_starred = preprocess_text(
        text,
        romanize,
        language_code,
        split_size,
        star_frequency,
    )

    segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)

    return postprocess_results(text_starred, spans, stride, scores, merge_threshold)


@app.post("/align", response_model=AlignmentResponse)
async def align(
    audio: UploadFile = File(..., description="Audio file to align"),
    text: str = Form(..., description="Transcript text to align with the audio"),
    language: str = Form("en", description="Language code: en/zh or ISO-639-3"),
    romanize: bool = Form(True, description="Apply uroman normalization"),
    split_size: str = Form("word", description="sentence, word, or char"),
    star_frequency: str = Form("edges", description="segment or edges"),
    merge_threshold: float = Form(0.0, description="Merge short gaps between segments"),
    batch_size: Optional[int] = Form(None, description="Override batch size per request"),
    window_size: Optional[float] = Form(None, description="Override chunk window size in seconds"),
    context_size: Optional[float] = Form(None, description="Override chunk context size in seconds"),
) -> AlignmentResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if split_size not in {"sentence", "word", "char"}:
        raise HTTPException(status_code=400, detail="split_size must be sentence, word, or char")
    if star_frequency not in {"segment", "edges"}:
        raise HTTPException(status_code=400, detail="star_frequency must be segment or edges")

    try:
        language_code = _resolve_language_code(language)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    req_batch_size = batch_size if batch_size is not None else RUNTIME_CONFIG["batch_size"]
    req_window_size = window_size if window_size is not None else RUNTIME_CONFIG["window_size"]
    req_context_size = context_size if context_size is not None else RUNTIME_CONFIG["context_size"]

    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        async with app.state.semaphore:
            loop = asyncio.get_running_loop()
            run_alignment = partial(
                _run_alignment,
                temp_path,
                text,
                language_code,
                romanize,
                split_size,
                star_frequency,
                merge_threshold,
                max(1, int(req_batch_size)),
                float(req_window_size),
                float(req_context_size),
            )
            segments = await loop.run_in_executor(app.state.executor, run_alignment)

        return AlignmentResponse(
            text=text,
            language=language_code,
            segments=segments,
            model=RUNTIME_CONFIG["alignment_model"],
            device=RUNTIME_CONFIG["device"],
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"alignment failed: {error}") from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await audio.close()
