"""Image retrieval and normalization helpers for Story Schema Cards."""

from __future__ import annotations

import hashlib
import io
import re
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


OPENVERSE_API_URL = "https://api.openverse.org/v1/images/"
OPEN_LICENSES = {"pdm", "cc0", "by"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_EDGE = 1200
REQUEST_TIMEOUT = (5, 30)


def _terms(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2
    }


def card_search_text(card: dict[str, Any]) -> str:
    return " ".join(
        str(card.get(field, ""))
        for field in (
            "label",
            "card_type",
            "description",
            "image_search_query",
            "generic_image_fallback",
        )
    )


def similarity_score(query: Any, candidate: Any) -> float:
    """Return a conservative token-overlap score between zero and one."""
    query_terms = _terms(query)
    candidate_terms = _terms(candidate)
    if not query_terms or not candidate_terms:
        return 0.0

    overlap = query_terms & candidate_terms
    coverage = len(overlap) / len(query_terms)
    precision = len(overlap) / len(candidate_terms)
    return (coverage * 0.75) + (precision * 0.25)


def choose_existing_asset(
    card: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    threshold: float = 0.72,
) -> dict[str, Any] | None:
    """Choose only a high-confidence stored image match."""
    query = card_search_text(card)
    card_type = str(card.get("card_type", "")).lower()
    best: dict[str, Any] | None = None
    best_score = 0.0

    for candidate in candidates:
        if not candidate.get("image_storage_path"):
            continue
        status = str(candidate.get("image_status", ""))
        if "placeholder" in status:
            continue

        candidate_text = " ".join(
            str(candidate.get(field, ""))
            for field in ("front_text", "category", "image_prompt")
        )
        score = similarity_score(query, candidate_text)
        if card_type and card_type == str(candidate.get("category", "")).lower():
            score += 0.08

        if score > best_score:
            best_score = score
            best = dict(candidate)

    if not best or best_score < threshold:
        return None

    best["match_score"] = round(min(best_score, 1.0), 3)
    return best


def search_openverse(
    card: dict[str, Any],
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Return the highest-relevance public-domain/CC0/CC BY image."""
    query = (
        card.get("image_search_query")
        or card.get("generic_image_fallback")
        or card.get("label")
        or "public domain book illustration"
    )
    client = session or requests
    response = client.get(
        OPENVERSE_API_URL,
        params={
            "q": str(query)[:200],
            "license": ",".join(sorted(OPEN_LICENSES)),
            "mature": "false",
            "page_size": 10,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    ranked_results: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        license_name = str(result.get("license", "")).lower()
        thumbnail = result.get("thumbnail")
        if result.get("mature") or license_name not in OPEN_LICENSES or not thumbnail:
            continue

        tags = " ".join(
            str(tag.get("name", "")) if isinstance(tag, dict) else str(tag)
            for tag in result.get("tags", [])
        )
        result_text = f"{result.get('title', '')} {tags}"
        score = max(
            similarity_score(query, result_text),
            similarity_score(card.get("label", ""), result_text),
        )
        ranked_results.append((score, result))

    if not ranked_results:
        return None

    score, result = max(ranked_results, key=lambda item: item[0])
    if score < 0.16:
        return None

    license_name = str(result.get("license", "")).lower()
    return {
            "source_type": "openverse",
            "source_provider": result.get("provider") or result.get("source") or "Openverse",
            "source_id": result.get("id"),
            "source_url": result.get("foreign_landing_url") or result.get("url"),
            "download_url": thumbnail,
            "title": result.get("title") or "Open-license image",
            "creator": result.get("creator") or "Unknown creator",
            "creator_url": result.get("creator_url"),
            "license": license_name,
            "license_url": result.get("license_url"),
            "attribution": result.get("attribution") or "",
            "match_score": round(score, 3),
    }


def download_image(url: str, session: requests.Session | None = None) -> bytes:
    """Download a bounded HTTP(S) image and normalize it for deck use."""
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Image URL must use HTTP or HTTPS.")

    client = session or requests
    response = client.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if not content_type.startswith("image/"):
        raise ValueError("The selected URL did not return an image.")

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_IMAGE_BYTES:
        raise ValueError("The selected image is larger than the 12 MB safety limit.")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError("The selected image is larger than the 12 MB safety limit.")
        chunks.append(chunk)

    return normalize_image(b"".join(chunks))


def normalize_image(image_bytes: bytes, max_edge: int = MAX_IMAGE_EDGE) -> bytes:
    """Apply orientation, RGB conversion, resizing, and JPEG compression."""
    if not image_bytes:
        raise ValueError("Image data is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image data is larger than the 12 MB safety limit.")

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True, progressive=True)
        return output.getvalue()


def crop_image(
    image_bytes: bytes,
    focal_x: int = 50,
    focal_y: int = 50,
    aspect_ratio: float = 1.65,
) -> bytes:
    """Crop an image to the card frame while keeping a user-selected focal point."""
    if not 0 <= focal_x <= 100 or not 0 <= focal_y <= 100:
        raise ValueError("Focal point values must be between 0 and 100.")

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        current_ratio = width / height

        if current_ratio > aspect_ratio:
            crop_height = height
            crop_width = int(height * aspect_ratio)
        else:
            crop_width = width
            crop_height = int(width / aspect_ratio)

        center_x = width * (focal_x / 100)
        center_y = height * (focal_y / 100)
        left = max(0, min(int(center_x - crop_width / 2), width - crop_width))
        top = max(0, min(int(center_y - crop_height / 2), height - crop_height))
        image = image.crop((left, top, left + crop_width, top + crop_height))

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=84, optimize=True, progressive=True)
        return output.getvalue()


def image_digest(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()
