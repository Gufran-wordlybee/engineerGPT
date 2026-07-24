"""Phase 2: grounded answer generation from routed textbook sections."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from config.settings import (
    BOOKS_PROCESSED_PATH,
    GENERATE_LLM_MODEL,
    GENERATE_LLM_MODEL_API,
    GENERATOR_LLM_TIMEOUT_SECONDS,
    GENERATOR_MAX_CONTEXT_CHARS,
    GENERATOR_MAX_TOKENS,
    GENERATOR_TEMPERATURE,
    VISION_MAX_IMAGE_SIZE_MB,
    VISION_MAX_IMAGES_PER_REQUEST,
    VISION_MODEL_NAME,
)

logger = logging.getLogger(__name__)
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_VISUAL_WORDS = {
    "diagram", "figure", "circuit", "waveform", "timing", "block diagram",
    "draw", "label", "show", "illustrate", "equation", "derive",
}


def load_section(book_name: str, section_id: str) -> dict:
    """Load one processed section, with the same clear failure style as router."""
    path = BOOKS_PROCESSED_PATH / book_name / "sections" / f"{section_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No section '{section_id}' found for book '{book_name}' at {path}."
        )
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_sections(book_name: str, section_ids: list[str]) -> list[dict]:
    """Load routed sections in order; one bad section must not lose the answer."""
    sections = []
    for section_id in section_ids:
        try:
            sections.append(load_section(book_name, section_id))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping routed section %s: %s", section_id, exc)
    return sections


def resolve_image_paths(book_name: str, section: dict) -> list[dict]:
    """Return existing visual files, rejecting paths outside the processed book."""
    root = (BOOKS_PROCESSED_PATH / book_name).resolve()
    visuals = []
    for kind, entries in (("figure", section.get("images", [])), ("equation", section.get("equations", []))):
        for entry in entries:
            relative = entry.get("path", "")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                logger.warning("Skipping unsafe visual path: %s", relative)
                continue
            if not path.is_file():
                logger.warning("Skipping missing visual file: %s", path)
                continue
            visuals.append({
                "path": str(path),
                "relative_path": relative,
                "type": entry.get("type", kind),
                "caption": entry.get("caption") or entry.get("label") or "",
            })
    return visuals


def build_prompt(question: str, sections: list[dict]) -> str:
    """Build the text portion of the multimodal request from routed sections."""
    remaining = GENERATOR_MAX_CONTEXT_CHARS
    blocks = []
    for section in sections:
        text = section.get("text", "").strip()
        # Router order is relevance order, so only later sections are shortened.
        if len(text) > remaining:
            text = text[:max(remaining, 0)].rsplit(" ", 1)[0] + "..."
        remaining -= len(text)
        blocks.append(
            f"[Section: {section.get('chapter', '')} -> {section.get('title', '')}]\n{text}"
        )
        if remaining <= 0:
            break
    context = "\n\n---\n\n".join(blocks)

    return f"""You are an exam-focused study assistant for engineering students.

STUDENT QUESTION: {question}

RELEVANT TEXTBOOK SECTION(S):
{context}

INSTRUCTIONS:
1. Answer directly and specifically for exam preparation.
2. Ground every claim in the supplied textbook sections; do not invent facts.
3. If diagrams or equations are attached, reference them only when relevant.
4. If the sections do not fully answer the question, say so plainly.
5. Structure the answer as concept, mechanism or derivation, then a short source-based example when available.
"""


def should_attach_images(question: str, sections: list[dict]) -> bool:
    """Use visuals when present; a visual question makes that decision explicit."""
    has_visuals = any(section.get("images") or section.get("equations") for section in sections)
    if not has_visuals:
        return False
    question_lower = question.lower()
    return any(word in question_lower for word in _VISUAL_WORDS) or has_visuals


def encode_image_b64(path: str) -> str:
    """Encode a locally extracted image for Groq's data-URL image format."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def _final_answer(content: str | None) -> str | None:
    """Drop Qwen reasoning blocks; only final answer text may reach the UI."""
    if not content:
        return None
    answer = re.sub(r"^\s*<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    # A response cut off while reasoning has no safe answer to display.
    return None if answer.lstrip().startswith("<think>") else answer.strip() or None


def _select_visuals(question: str, book_name: str, sections: list[dict]) -> list[dict]:
    """Keep Groq requests under image count and per-image size limits."""
    computational = any(word in question.lower() for word in ("equation", "derive", "calculate", "solve"))
    visuals = [visual for section in sections for visual in resolve_image_paths(book_name, section)]
    if computational:
        visuals.sort(key=lambda visual: visual["type"] != "equation")
    accepted = []
    max_bytes = VISION_MAX_IMAGE_SIZE_MB * 1024 * 1024
    for visual in visuals:
        if Path(visual["path"]).stat().st_size > max_bytes:
            logger.warning("Skipping oversized visual: %s", visual["relative_path"])
            continue
        accepted.append(visual)
        if len(accepted) == VISION_MAX_IMAGES_PER_REQUEST:
            break
    if len(visuals) > len(accepted):
        logger.info("Generator attached %d of %d candidate visuals.", len(accepted), len(visuals))
    return accepted


def _call_generator_llm(prompt: str, visuals: list[dict]) -> str | None:
    """Call Qwen through Groq's OpenAI-compatible chat endpoint."""
    if not GENERATE_LLM_MODEL_API:
        logger.error("Set GENERATE_LLM_Model_API or GROQ_LLM_API_KEY in .env.")
        return None
    try:
        from openai import OpenAI

        content: list[dict] = [{"type": "text", "text": prompt}]
        for visual in visuals:
            encoded = encode_image_b64(visual["path"])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            })
        client = OpenAI(
            api_key=GENERATE_LLM_MODEL_API,
            base_url=_GROQ_BASE_URL,
            timeout=GENERATOR_LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME if visuals else GENERATE_LLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=GENERATOR_TEMPERATURE,
            max_tokens=GENERATOR_MAX_TOKENS,
            # The installed OpenAI SDK forwards Groq-only options via extra_body.
            # Qwen's non-thinking mode keeps private reasoning out of answers.
            extra_body={"reasoning_effort": "none", "reasoning_format": "hidden"},
        )
        return _final_answer(response.choices[0].message.content)
    except Exception as exc:
        logger.warning("Generator LLM call failed: %s", exc)
        return None


def generate_answer(
    question: str,
    book_name: str,
    section_ids: list[str],
    force_vision: bool | None = None,
) -> dict:
    """Generate one grounded answer; return a display-safe error instead of raising."""
    result = {
        "answer": "",
        "sections_used": [],
        "images_used": [],
        "model": GENERATE_LLM_MODEL,
        "used_vision": False,
    }
    try:
        sections = load_sections(book_name, section_ids)
        if not sections:
            result["answer"] = "I could not load the textbook sections selected for this question."
            return result
        use_vision = force_vision if force_vision is not None else should_attach_images(question, sections)
        visuals = _select_visuals(question, book_name, sections) if use_vision else []
        answer = _call_generator_llm(build_prompt(question, sections), visuals)
        if not answer:
            result["answer"] = "I could not generate an answer right now. Please check the generator API key and model name."
            return result
        result.update({
            "answer": answer,
            "sections_used": [section["section_id"] for section in sections],
            "images_used": [visual["relative_path"] for visual in visuals],
            "model": VISION_MODEL_NAME if visuals else GENERATE_LLM_MODEL,
            "used_vision": bool(visuals),
        })
        return result
    except Exception as exc:
        logger.exception("Answer generation failed: %s", exc)
        result["answer"] = "I could not prepare an answer from the selected textbook sections."
        return result


if __name__ == "__main__":
    assert should_attach_images("show the circuit diagram", [{"images": [{}]}])
    assert not should_attach_images("define cache", [{"images": [], "equations": []}])
    assert _final_answer("<think>private reasoning</think>Final answer") == "Final answer"
