"""
figure_extractor.py - Shared utility for figure/caption extraction and saving.

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 26-02-2025
"""

# Standard library imports
import os
import json
import shutil
import re
from io import BytesIO


class FigureExtractor:
    """
    Utility class for extracting figures from articles and saving them alongside
    their captions. Used by all article processors (Elsevier, Springer, IOP, Wiley, PDF).

    Output structure:
        {base_path}/{doi_}/{caption_id}.jpg
        {base_path}/{doi_}/info.json  →  {caption_id: caption_text, ...}

    The default base_path is "results/related_figures". Callers may pass an explicit
    base_path to organise figures under a different root (e.g.
    "results/extracted_data/{property_name}/related_figures").
    """

    BASE_PATH = "results/related_figures"

    @staticmethod
    def keyword_matches_caption(caption_text: str, main_figure_keywords: dict) -> bool:
        """
        Check whether the caption text contains any of the given keywords.

        Args:
            caption_text (str): The figure caption text to check.
            main_figure_keywords (dict): Dictionary with optional keys:
                - "exact_keywords": List of keywords for case-insensitive substring match.
                - "substring_keywords": List of strings for case-insensitive substring match.

        Returns:
            bool: True if any keyword matches, False otherwise.
        """
        if not caption_text or not main_figure_keywords:
            return False

        text_lower = caption_text.lower()
        # Space-normalized text keeps token boundaries while ignoring punctuation.
        text_spaced = re.sub(r"[^a-z0-9]+", " ", text_lower).strip()
        # Compact text removes all separators to match forms like d33 vs d 33.
        text_compact = re.sub(r"[^a-z0-9]+", "", text_lower)

        for kw in main_figure_keywords.get("exact_keywords", []):
            kw_lower = kw.lower()
            kw_compact = re.sub(r"[^a-z0-9]+", "", kw_lower)
            if kw_compact and kw_compact in text_compact:
                return True

        for kw in main_figure_keywords.get("substring_keywords", []):
            kw_spaced = re.sub(r"[^a-z0-9]+", " ", kw.lower()).strip()
            if kw_spaced and kw_spaced in text_spaced:
                return True

        return False

    @staticmethod
    def doi_to_folder_name(doi: str) -> str:
        """Replace '/' with '_' in a DOI to make it filesystem-safe."""
        return doi.replace("/", "_")

    @classmethod
    def get_figure_dir(cls, doi: str, base_path: str = None) -> str:
        """Return the directory path for a given DOI's figures.

        Args:
            doi (str): Article DOI.
            base_path (str, optional): Override the default BASE_PATH root.
        """
        root = base_path if base_path is not None else cls.BASE_PATH
        return os.path.join(root, cls.doi_to_folder_name(doi))

    @classmethod
    def save_figure_from_bytes(
        cls, image_bytes: bytes, doi: str, caption_id: str, base_path: str = None
    ) -> str:
        """
        Save raw image bytes to {base_path}/{doi_}/{caption_id}.jpg.

        Args:
            image_bytes (bytes): Raw image data.
            doi (str): Article DOI.
            caption_id (str): Caption identifier (e.g., "gr1", "Fig1", "figure_0").
            base_path (str, optional): Override the default BASE_PATH root.

        Returns:
            str: The saved file path, or None if saving failed.
        """
        try:
            fig_dir = cls.get_figure_dir(doi, base_path)
            os.makedirs(fig_dir, exist_ok=True)
            # Sanitize caption_id for use as filename
            safe_id = _sanitize_filename(caption_id)
            out_path = os.path.join(fig_dir, f"{safe_id}.jpg")
            # Convert image to JPEG via PIL; handles PNG, GIF, WEBP, etc.
            try:
                from PIL import Image

                img = Image.open(BytesIO(image_bytes))
                # For animated GIF/WEBP, use the first frame
                try:
                    img.seek(0)
                except (AttributeError, EOFError):
                    pass
                # Composite transparent images (RGBA/LA/palette) onto white background
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode == "P":
                    img = img.convert("RGBA")
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
                img.save(out_path, "JPEG")
            except Exception:
                # Fallback: write bytes as-is
                with open(out_path, "wb") as f:
                    f.write(image_bytes)
            return out_path
        except Exception:
            return None

    @classmethod
    def save_figure_from_local_path(
        cls, src_path: str, doi: str, caption_id: str, base_path: str = None
    ) -> str:
        """
        Copy a local image file to {base_path}/{doi_}/{caption_id}.jpg.

        Args:
            src_path (str): Path to the source image file.
            doi (str): Article DOI.
            caption_id (str): Caption identifier.
            base_path (str, optional): Override the default BASE_PATH root.

        Returns:
            str: The saved file path, or None if file not found or copy failed.
        """
        if not src_path or not os.path.isfile(src_path):
            return None
        try:
            with open(src_path, "rb") as f:
                image_bytes = f.read()
            return cls.save_figure_from_bytes(image_bytes, doi, caption_id, base_path)
        except Exception:
            return None

    @classmethod
    def update_info_json(
        cls, doi: str, caption_id: str, caption_text: str, base_path: str = None
    ):
        """
        Add or update an entry in {base_path}/{doi_}/info.json.
        The JSON maps caption_id → caption_text string.

        Args:
            doi (str): Article DOI.
            caption_id (str): Caption identifier.
            caption_text (str): The full caption text.
            base_path (str, optional): Override the default BASE_PATH root.
        """
        try:
            fig_dir = cls.get_figure_dir(doi, base_path)
            os.makedirs(fig_dir, exist_ok=True)
            info_path = os.path.join(fig_dir, "info.json")
            if os.path.isfile(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            else:
                info = {}
            safe_id = _sanitize_filename(caption_id)
            info[safe_id] = caption_text
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def record_failed_article(
    doi: str,
    publisher: str,
    reason: str,
    report_path: str,
    enabled: bool = True,
) -> None:
    """Append a tab-separated (doi, publisher, reason) entry to the automated failure report.

    Args:
        doi (str): Article DOI.
        publisher (str): Publisher name (e.g. "elsevier", "springer").
        reason (str): Short failure reason (e.g. "download_failed", "xml_parse_failed").
        report_path (str): Path to the report file.
        enabled (bool): When False, the function is a no-op. Defaults to True.
    """
    if not enabled:
        return
    entry = f"{doi}\t{publisher}\t{reason}"
    try:
        dir_name = os.path.dirname(report_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def _sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe for filenames."""
    unsafe = r'\/:*?"<>|'
    for ch in unsafe:
        name = name.replace(ch, "_")
    return name.strip("._").replace(" ", "_")
