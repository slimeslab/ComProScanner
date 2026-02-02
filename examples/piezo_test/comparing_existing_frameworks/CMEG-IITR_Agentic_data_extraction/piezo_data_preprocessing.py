"""
piezo_data_preprocessing.py

Processes Elsevier XML files for piezoelectric materials.
Extracts fulltext and tables into DOI-named folders.
"""

import os
import json
import unicodedata
import re
from lxml import etree
import csv

# === Constants ===
INPUT_XML_DIR = "Elsevier_xml_data"
OUTPUT_ROOT_DIR = "elsevier_piezo_processed"

# === Namespaces ===
ns = {
    "ce": "http://www.elsevier.com/xml/common/elssce",
    "xocs": "http://www.elsevier.com/xml/xocs/dtd",
    "dc": "http://purl.org/dc/elements/1.1/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
}


# === Text Cleaning ===
def clean(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text.strip())
    return re.sub(r"\s+", " ", text)


def extract_texts(elem):
    return clean(" ".join(t.strip() for t in elem.itertext() if t.strip()))


def clean_caption_by_removing_row_text(caption, rows):
    caption_cleaned = caption
    for row in rows:
        row_text = " ".join(row)
        if row_text in caption_cleaned:
            caption_cleaned = caption_cleaned.replace(row_text, "")
    return clean(caption_cleaned)


# === Full Text Extraction ===
def extract_elsevier_article(xml_file_path):
    try:
        tree = etree.parse(xml_file_path)
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return None

    # Metadata
    doi_element = root.find(".//prism:doi", namespaces=ns)
    doi = clean(doi_element.text) if doi_element is not None else "N/A"

    title_element = root.find(".//dc:title", namespaces=ns)
    title = clean(title_element.text) if title_element is not None else "N/A"

    abstract_element = root.find(".//dc:description", namespaces=ns)
    abstract = clean(abstract_element.text) if abstract_element is not None else "N/A"

    # Sequential Section Parsing
    sections = {}
    current_section = "Introduction"
    sections[current_section] = []

    for elem in root.iter():
        tag = etree.QName(elem.tag).localname

        if tag == "section-title":
            section_title_text = extract_texts(elem)
            if section_title_text:
                current_section = section_title_text
                if current_section not in sections:
                    sections[current_section] = []

        elif tag == "para":
            para_text = extract_texts(elem)
            if para_text:
                sections.setdefault(current_section, []).append(para_text)
        elif tag == "abstract":
            if abstract == "N/A":
                abstract = extract_texts(elem)

    return {"doi": doi, "title": title, "abstract": abstract, "sections": sections}


# === Table Extraction ===
def extract_elsevier_tables_from_xml(xml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return []

    caption_pattern = re.compile(r"\b(?:table|able)\s*\d+(?:[\.:])?", re.IGNORECASE)

    captions = []
    tables_data = []

    current_caption = None
    current_table_rows = []
    skip_elem_id = None

    for elem in root.iter():
        text = extract_texts(elem)

        if caption_pattern.match(text) and len(text.split()) > 3:
            if current_caption and current_table_rows:
                captions.append(current_caption)
                tables_data.append(current_table_rows)

            current_caption = text
            current_table_rows = []
            skip_elem_id = id(elem)
            continue

        if current_caption is not None and id(elem) != skip_elem_id:
            row_values = []
            for child in elem:
                tag = etree.QName(child.tag).localname.lower()
                if tag in {"entry", "td", "cell", "data"}:
                    value = extract_texts(child)
                    if value:
                        row_values.append(value)
            if row_values:
                if not any(re.search(r"\d", val) for val in row_values):
                    continue
                current_table_rows.append(row_values)

    if current_caption and current_table_rows:
        captions.append(current_caption)
        tables_data.append(current_table_rows)

    saved_files_info = []
    for i, (caption, rows) in enumerate(zip(captions, tables_data), 1):
        cleaned_caption = clean_caption_by_removing_row_text(caption, rows)

        caption_file = os.path.join(output_dir, f"table{i}_caption.txt")
        csv_file = os.path.join(output_dir, f"table{i}.csv")

        try:
            with open(caption_file, "w", encoding="utf-8") as f:
                f.write(cleaned_caption)
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if rows:
                    writer.writerows(rows)
            saved_files_info.append((caption_file, csv_file))
        except IOError as e:
            print(f"Error saving files: {e}")
            continue

    return saved_files_info


if __name__ == "__main__":
    os.makedirs(INPUT_XML_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    xml_files = [f for f in os.listdir(INPUT_XML_DIR) if f.endswith(".xml")]

    if not xml_files:
        print(
            f"No XML files found in '{INPUT_XML_DIR}'. Please place your XML files there."
        )
    else:
        print(
            f"Found {len(xml_files)} XML files in '{INPUT_XML_DIR}'. Starting processing..."
        )

        for xml_filename in xml_files:
            xml_file_path = os.path.join(INPUT_XML_DIR, xml_filename)
            print(f"\nProcessing '{xml_filename}'...")

            article_data = extract_elsevier_article(xml_file_path)

            if article_data and article_data["doi"] != "N/A":
                doi_folder_name = re.sub(r"[^\w\-_.]", "_", article_data["doi"])
                article_output_dir = os.path.join(OUTPUT_ROOT_DIR, doi_folder_name)
                os.makedirs(article_output_dir, exist_ok=True)

                # Save fulltext
                fulltext_path = os.path.join(article_output_dir, "fulltext.txt")
                try:
                    with open(fulltext_path, "w", encoding="utf-8") as f:
                        f.write(f"Title: {article_data['title']}\n\n")
                        f.write(f"DOI: {article_data['doi']}\n\n")
                        f.write(f"Abstract:\n{article_data['abstract']}\n\n")
                        for section, paras in article_data["sections"].items():
                            f.write(f"\n=== {section} ===\n")
                            for para in paras:
                                f.write(f"{para}\n\n")
                    print(f"✅ Full text saved to: {fulltext_path}")

                    # Save token count
                    token_count = len(
                        open(fulltext_path, "r", encoding="utf-8").read().split()
                    )
                    with open(
                        os.path.join(article_output_dir, "token_count.txt"), "w"
                    ) as f:
                        f.write(str(token_count))

                except IOError as e:
                    print(f"Error saving full text: {e}")

                # Extract tables
                table_info = extract_elsevier_tables_from_xml(
                    xml_file_path, article_output_dir
                )
                print(f"✅ Extracted {len(table_info)} tables")

            else:
                print(
                    f"❌ Failed to extract article data or DOI from '{xml_filename}'. Skipping processing."
                )

        print("\n🎉 Processing complete.")
