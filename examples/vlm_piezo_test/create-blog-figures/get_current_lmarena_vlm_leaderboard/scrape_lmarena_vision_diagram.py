#!/usr/bin/env python3
"""Scrape Arena vision leaderboard and join with LiteLLM token pricing."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


LEADERBOARD_URL = "https://arena.ai/leaderboard/vision/diagram"
LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
OUT_FILE = "vlm_model_score_price.csv"


COMPANY_PROVIDER_ALIASES: dict[str, list[str]] = {
    "alibaba": ["alibaba", "qwen"],
    "anthropic": ["anthropic"],
    "baidu": ["baidu"],
    "bytedance": ["bytedance", "volcengine"],
    "google": ["google", "vertex_ai", "gemini"],
    "mistral": ["mistral", "mistral_ai"],
    "openai": ["openai"],
    "stepfun": ["stepfun"],
    "tencent": ["tencent"],
    "xai": ["xai", "x-ai"],
    "x-ai": ["xai", "x-ai"],
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9._:/-]", "", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value


def extract_company(provider_text: str) -> str:
    text = provider_text.replace("Â·", "·")
    if "·" in text:
        return text.split("·", 1)[0].strip()
    return text.strip()


def parse_elo_score(score_cell_text: str) -> str:
    # Handles values like "1302 ±12" or "1302 Â±12".
    m = re.search(r"\b(\d{3,4})\b", score_cell_text)
    return m.group(1) if m else ""


def parse_rows_from_soup(soup: BeautifulSoup) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        # First column is rank; skip non-data rows.
        rank = tds[0].get_text(" ", strip=True)
        if not rank.isdigit():
            continue

        model_anchor = tds[2].find("a", attrs={"title": True})
        if model_anchor is None:
            continue

        model = (model_anchor.get("title") or "").strip() or model_anchor.get_text(
            " ", strip=True
        )
        if not model or model in seen:
            continue

        provider_span = tds[2].find(
            "span", string=lambda s: isinstance(s, str) and ("·" in s or "Â·" in s)
        )
        if provider_span is None:
            continue

        company = extract_company(provider_span.get_text(" ", strip=True))
        elo_score = parse_elo_score(tds[3].get_text(" ", strip=True))
        if not company or not elo_score:
            continue

        rows.append(
            {
                "model": model,
                "company": company,
                "elo_score": elo_score,
                "input_price_per_1m": "",
                "output_price_per_1m": "",
            }
        )
        seen.add(model)

    if not rows:
        raise RuntimeError("Could not parse leaderboard rows from HTML.")
    return rows


def fetch_live_soup() -> BeautifulSoup:
    response = requests.get(
        LEADERBOARD_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; lmarena-scraper/1.0)"},
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def fetch_leaderboard_rows() -> list[dict[str, str]]:
    return parse_rows_from_soup(fetch_live_soup())


def parse_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def fmt_price(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def fetch_litellm_prices() -> dict[str, dict[str, str]]:
    response = requests.get(
        LITELLM_PRICES_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; lmarena-scraper/1.0)"},
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        return {}

    prices: dict[str, dict[str, str]] = {}

    for model_key, spec in data.items():
        if model_key == "sample_spec" or not isinstance(spec, dict):
            continue

        in_token = parse_float(spec.get("input_cost_per_token"))
        out_token = parse_float(spec.get("output_cost_per_token"))
        if in_token is None and out_token is None:
            continue

        in_1m = in_token * 1_000_000 if in_token is not None else None
        out_1m = out_token * 1_000_000 if out_token is not None else None

        normalized_key = slugify(str(model_key))
        prices[normalized_key] = {
            "input": fmt_price(in_1m),
            "output": fmt_price(out_1m),
        }

        if "/" in normalized_key:
            bare = normalized_key.split("/", 1)[1]
            prices.setdefault(
                bare, {"input": fmt_price(in_1m), "output": fmt_price(out_1m)}
            )

    return prices


def build_candidate_keys(model: str, company: str) -> list[str]:
    model_slug = slugify(model)
    model_variants = [model_slug]

    # Strip trailing date-like suffixes and '-preview' for fallback matching.
    stripped = re.sub(r"[-_]?20\d{6,8}$", "", model_slug)
    stripped = re.sub(r"[-_]?preview$", "", stripped)
    stripped = stripped.strip("-_")
    if stripped and stripped not in model_variants:
        model_variants.append(stripped)

    company_slug = slugify(company)
    providers = COMPANY_PROVIDER_ALIASES.get(
        company_slug, [company_slug] if company_slug else []
    )

    keys: list[str] = []
    for mv in model_variants:
        keys.append(mv)
        for provider in providers:
            keys.append(f"{provider}/{mv}")

    unique: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def attach_prices(
    rows: list[dict[str, str]],
    litellm_prices: dict[str, dict[str, str]],
) -> None:
    for row in rows:
        candidates = build_candidate_keys(row["model"], row["company"])
        matched = None
        for key in candidates:
            matched = litellm_prices.get(key)
            if matched is not None:
                break

        if matched is None:
            continue

        row["input_price_per_1m"] = matched.get("input", "")
        row["output_price_per_1m"] = matched.get("output", "")


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "company",
                "elo_score",
                "input_price_per_1m",
                "output_price_per_1m",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent

    rows = fetch_leaderboard_rows()
    litellm_prices = fetch_litellm_prices()
    attach_prices(rows, litellm_prices)

    out_path = here / OUT_FILE
    write_csv(rows, out_path)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
