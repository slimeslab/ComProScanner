"""
iop_processor.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 10-03-2025
"""

# Standard library imports
import os
import time
import re
import requests

# Third party imports
import pandas as pd
from lxml import etree
from tqdm import tqdm
from sqlalchemy import create_engine

# Custom imports
from ..utils.configs import (
    DatabaseConfig,
    DefaultPaths,
    ArticleRelatedKeywords,
    RAGConfig,
)
from ..utils.database_manager import (
    MySQLDatabaseManager,
    CSVDatabaseManager,
    VectorDatabaseManager,
)
from ..utils.error_handler import ValueErrorHandler, KeyboardInterruptHandler
from ..utils.logger import setup_logger
from ..utils.common_functions import return_error_message
from ..utils.prepare_iop_files import PrepareIOPFiles
from ..utils.figure_extractor import FigureExtractor, record_failed_article


# configure logger
logger = setup_logger("comproscanner.log", module_name="iop_processor")


######## Class to process IOP articles ########
class IOPArticleProcessor:
    """
    Get the article data from the stored XML files and process them to extract the required sections of the article and write them to the database and CSV files.

    Args:
        main_property_keyword (str: Required): The main keyword to process the articles for and file naming
        property_keywords (dict: Required): A dictionary of property keywords which will be used for filtering sentences and should look like the following:
        {
            "exact_keywords": ["example1", "example2"],
            "substring_keywords": [" example 1 ", " example 2 "],
        }
        sql_batch_size (int): The number of rows to write to the database at once (Applicable only if is_sql_db is True) (default: 500)
        csv_batch_size (int): The number of rows to write to the CSV file at once (default: 1)
        start_row (int): The row number to start processing from (default: None)
        end_row (int): The row number to end processing at (default: None)
        doi_list (list): A list of DOIs to process (default: None)
        is_sql_db (bool): A flag to indicate if the data should be written to the database (default: False)
        rag_config (RAGConfig): An instance of the RAGConfig class (default: RAGConfig())
    """

    def __init__(
        self,
        main_property_keyword: str = None,
        property_keywords: dict = None,
        sql_batch_size: int = 500,
        csv_batch_size: int = 1,
        start_row: int = None,
        end_row: int = None,
        doi_list: list = None,
        is_sql_db: bool = False,
        rag_config: RAGConfig = RAGConfig(),
        caption_keywords: dict = None,
        save_failed_automated_report: bool = True,
        failed_automated_report_path: str = None,
    ):
        keyword_message = return_error_message("main_property_keyword")
        property_keywords_message = return_error_message("property_keywords")
        # Required parameters
        self.keyword = main_property_keyword
        if self.keyword is None:
            logger.error(f"{keyword_message}")
            raise ValueErrorHandler(f"{keyword_message}")
        self.property_keywords = property_keywords
        if self.property_keywords is None:
            logger.error(f"{property_keywords_message}")
            raise ValueErrorHandler(f"{property_keywords_message}")
        # create instances
        self.all_paths = DefaultPaths(self.keyword)
        self.db_configs = DatabaseConfig(self.keyword, is_sql_db)
        self.metadata_csv_filename = self.all_paths.METADATA_CSV_FILENAME
        self.csv_path = self.db_configs.EXTRACTED_CSV_FOLDERPATH
        self.paperdata_table_name = self.db_configs.PAPERDATA_TABLE_NAME
        self.iop_folderpath = self.all_paths.IOP_FOLDERPATH
        self.sql_batch_size = sql_batch_size
        self.csv_batch_size = csv_batch_size
        self.prepare_iop_files = PrepareIOPFiles(self.keyword, logger)
        # Optional parameters
        self.start_row = start_row
        self.end_row = end_row
        self.doi_list = doi_list
        self.is_sql_db = is_sql_db
        self.rag_config = rag_config
        self.caption_keywords = caption_keywords
        # Takes from config file
        self.timeout_file = self.all_paths.TIMEOUT_DOI_LOG_FILENAME
        self.article_related_keywords = ArticleRelatedKeywords()

        self.df = None
        self.new_df = pd.DataFrame(
            columns=[
                "doi",
                "article_title",
                "publication_name",
                "publisher",
                "abstract",
                "introduction",
                "exp_methods",
                "comp_methods",
                "results_discussion",
                "conclusion",
                "is_property_mentioned",
            ]
        )
        self.valid_property_articles = 0
        self.source = "iop"
        self.csv_filepath = (
            f"{self.csv_path}/{self.source}_{self.keyword}_paragraphs.csv"
        )
        self.save_failed_automated_report = save_failed_automated_report
        self.failed_automated_report_path = (
            failed_automated_report_path or "results/failed_automated_articles.txt"
        )
        self.failed_automated_count = 0

        self.sql_db_manager = MySQLDatabaseManager(self.keyword, self.is_sql_db)
        self.csv_db_manager = CSVDatabaseManager()
        self.vector_db_manager = VectorDatabaseManager(rag_config=self.rag_config)
        self._iop_revision_cache = {}

    def _record_failed_article(self, doi: str, reason: str) -> None:
        """Record a failed article to the automated failure report."""
        self.failed_automated_count += 1
        record_failed_article(
            doi, self.source, reason,
            self.failed_automated_report_path,
            self.save_failed_automated_report,
        )

    def _get_first_xpath_text(self, root, xpath_expr: str) -> str:
        values = root.xpath(xpath_expr)
        for value in values:
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
            elif value is not None and hasattr(value, "text") and value.text:
                cleaned = value.text.strip()
                if cleaned:
                    return cleaned
        return ""

    def _get_iop_cdn_path_parts(self, root):
        issn = self._get_first_xpath_text(
            root,
            './/*[local-name()="journal-meta"]/*[local-name()="issn"][1]/text()',
        )
        volume = self._get_first_xpath_text(
            root,
            './/*[local-name()="article-meta"]/*[local-name()="volume"][1]/text()',
        )
        issue = self._get_first_xpath_text(
            root,
            './/*[local-name()="article-meta"]/*[local-name()="issue"][1]/text()',
        )
        elocation_id = self._get_first_xpath_text(
            root,
            './/*[local-name()="article-meta"]/*[local-name()="elocation-id"][1]/text()',
        )
        if not (issn and volume and issue and elocation_id):
            return None
        return issn, volume, issue, elocation_id

    def _is_iop_no_such_key(self, response: requests.Response) -> bool:
        if response is None:
            return False
        if response.status_code == 404:
            return True
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "xml" not in content_type:
            return False
        body = (response.text or "").strip()
        return "<Code>NoSuchKey</Code>" in body

    def _download_iop_figure_from_content_cdn(
        self,
        issn: str,
        volume: str,
        issue: str,
        elocation_id: str,
        figure_filename: str,
        cache_key: str,
    ):
        cached_revision = self._iop_revision_cache.get(cache_key)
        revision_candidates = [cached_revision] if cached_revision else []
        revision_candidates.extend(
            revision
            for revision in range(1, 10)
            if cached_revision is None or revision != cached_revision
        )
        for revision in revision_candidates:
            url = (
                "https://content.cld.iop.org/journals/"
                f"{issn}/{volume}/{issue}/{elocation_id}/revision{revision}/{figure_filename}"
            )
            try:
                resp = requests.get(url, timeout=20)
            except Exception as exc:
                logger.warning(f"Error calling IOP content CDN URL '{url}': {exc}")
                continue

            if resp.status_code == 200 and not self._is_iop_no_such_key(resp):
                self._iop_revision_cache[cache_key] = revision
                return resp.content

            if self._is_iop_no_such_key(resp):
                continue

            logger.warning(
                f"Unexpected response for IOP content CDN URL '{url}' "
                f"(HTTP {resp.status_code})."
            )
        return None

    def _load_and_preprocess_data(self):
        """
        Load and preprocess the metadata CSV file to get the DOIs of the articles to process.
        """
        self.df = pd.read_csv(self.metadata_csv_filename)
        self.df = self.df.dropna(subset=["doi"])

        # Apply row limits if specified
        if self.start_row is not None and self.end_row is not None:
            self.df = self.df.iloc[self.start_row : self.end_row]
        elif self.start_row is not None:
            self.df = self.df.iloc[self.start_row :]
        elif self.end_row is not None:
            self.df = self.df.iloc[: self.end_row]

        # Filter DOIs for the IOP publisher
        self.df = self.df[self.df["general_publisher"].str.lower() == "iop"]
        self.df = self.df.reset_index(drop=True)

        # Get processed DOIs from all sources
        processed_dois = set()

        # Create CSV directory if it doesn't exist
        os.makedirs(self.csv_path, exist_ok=True)

        # Check CSV files
        if os.path.exists(self.csv_filepath):
            try:
                df = pd.read_csv(self.csv_filepath)
                processed_dois.update(df["doi"].tolist())
            except Exception as e:
                logger.warning(
                    f"Error reading CSV file: {e}. Processed DOIs from CSV will be ignored."
                )

        # Check SQL database if enabled
        if self.is_sql_db:
            try:
                sql_engine = create_engine(self.db_configs.DATABASE_CONNECTION_URL)
                if self.sql_db_manager.table_exists(self.paperdata_table_name):
                    db_df = pd.read_sql_query(
                        f"SELECT doi FROM {self.paperdata_table_name}", sql_engine
                    )
                    processed_dois.update(db_df["doi"].tolist())
                sql_engine.dispose()
            except Exception as e:
                logger.warning(
                    f"Error reading database: {e}. Processed DOIs from database will be ignored."
                )

        unprocessed_dois = set(self.df["doi"]) - processed_dois
        self.df = self.df[self.df["doi"].isin(unprocessed_dois)]

    def _find_article(self, path: str):
        try:
            tree = etree.parse(path.strip())
            root = tree.getroot()
            body = root.xpath(f'.//*[local-name()="body"]')
            return root if body else None
        except etree.XMLSyntaxError as e:
            logger.error(f"Error parsing XML: {e}")
            return None

    def get_text(self, elements):
        return "".join(
            e.text.replace(" ", "") if e.text else "" for e in elements
        ).rstrip()

    def replace_element_with_text(self, element, text_to_add):
        parent = element.getparent()
        index = parent.index(element)
        if index > 0:
            target = parent[index - 1]
            target.tail = (
                (target.tail or "") + text_to_add.lstrip() + (element.tail or "")
            )
        else:
            parent.text = (
                (parent.text or "") + text_to_add.lstrip() + (element.tail or "")
            )
        parent.remove(element)

    def _modify_specific_element(self, sections, element_name):
        for element in sections.xpath(f'//*[local-name()="{element_name}"]'):
            if element_name == "math":
                mn_elements = element.xpath('.//*[local-name()="mn"]')
                sup_elements = element.xpath('.//*[local-name()="msup"]')
                if mn_elements and sup_elements:
                    mn_text = "".join(
                        f"E^{{{e.text.strip()}}}" if e.text else "" for e in mn_elements
                    )
                    other_text = self.get_text(
                        [e for e in element.iter() if e not in mn_elements]
                    )
                    text_to_add = other_text + mn_text
                else:
                    text_to_add = self.get_text(element.iter())
            elif element_name == "sup":
                text_to_add = f"E^{{{element.text}}}"
            self.replace_element_with_text(element, text_to_add)

    def _modify_all_elements(self, filtered_sections):
        self._modify_specific_element(filtered_sections, "sup")
        self._modify_specific_element(filtered_sections, "math")
        return filtered_sections

    def _process_xml(self, root):
        def _remove_elements(element_names, req_sections):
            for element_name in element_names:
                elements = req_sections.xpath(f'.//*[local-name()="{element_name}"]')
                if elements:
                    for element in elements:
                        parent = element.getparent()
                        index = parent.index(element)
                        if element.tail:
                            if index > 0:
                                preceding_sibling = parent[index - 1]
                                preceding_sibling.tail = (
                                    preceding_sibling.tail or ""
                                ) + element.tail
                            else:
                                parent.text = (parent.text or "") + element.tail
                        parent.remove(element)
            return req_sections

        req_sections = []
        modified_sections = []
        abstract = root.xpath('.//*[local-name()="abstract"]')
        all_sections = root.xpath("./sec")
        if all_sections:
            for section in all_sections:
                title_element = section.xpath('./child::*[local-name()="title"]')
                if title_element:
                    try:
                        title = " ".join([t.text for t in title_element]).lower()
                    except:
                        title = ""
                    if title:
                        if any(
                            word in title.lower()
                            for word_list in self.article_related_keywords.SECTION_TITLE_WORDS.values()
                            for word in word_list
                        ):
                            req_sections.append(section)
        element_names = ["inline-formula", "xref"]
        for section in req_sections:
            filtered_section = _remove_elements(element_names, section)
            modified_section = self._modify_all_elements(filtered_section)
            modified_sections.append(modified_section)
        if abstract:
            return abstract, modified_sections
        else:
            return None, modified_sections

    def _extract_and_save_figures(self, root, doi: str, xml_dir: str):
        """
        Extract figures from IOP JATS XML whose captions match self.caption_keywords.
        Tries local image files first (co-located with the XML), then IOP CDN as fallback.
        Saves to results/extracted_data/{keyword}/related_figures/{doi_}/.

        Args:
            root: lxml root element of the parsed IOP XML.
            doi (str): Article DOI.
            xml_dir (str): Directory containing the XML file (for local image lookup).
        """
        if not self.caption_keywords:
            return False
        base_path = f"results/extracted_data/{self.keyword}/related_figures"
        try:
            XLINK = "http://www.w3.org/1999/xlink"
            cdn_parts = self._get_iop_cdn_path_parts(root)
            cache_key = None
            if cdn_parts:
                issn, volume, issue, elocation_id = cdn_parts
                cache_key = f"{issn}/{volume}/{issue}/{elocation_id}"
            else:
                logger.warning(
                    f"Could not find ISSN/volume/issue/elocation-id for {doi}. "
                    "IOP content CDN figure fallback may fail."
                )
            figures = root.xpath('.//*[local-name()="fig"]')
            has_caption_keyword_match = False
            for i, figure in enumerate(figures):
                caption_id = figure.get("id", f"fig_{i}")

                label_els = figure.xpath('.//*[local-name()="label"]')
                label_text = (
                    "".join(label_els[0].itertext()).strip() if label_els else ""
                )
                caption_paras = figure.xpath(
                    './/*[local-name()="caption"]//*[local-name()="p"]'
                )
                para_text = " ".join(
                    "".join(p.itertext()) for p in caption_paras
                ).strip()
                caption_text = f"{label_text} {para_text}".strip()

                if not FigureExtractor.keyword_matches_caption(
                    caption_text, self.caption_keywords
                ):
                    continue

                has_caption_keyword_match = True
                FigureExtractor.update_info_json(
                    doi, caption_id, caption_text, base_path
                )

                graphic_els = figure.xpath('.//*[local-name()="graphic"]')
                href = None
                low_href = None
                high_href = None
                any_jpg_href = None
                for g in graphic_els:
                    candidate_href = g.get(f"{{{XLINK}}}href") or g.get("href")
                    if not candidate_href:
                        continue
                    content_type = (g.get("content-type") or "").lower()
                    if content_type == "low" and low_href is None:
                        low_href = candidate_href
                    elif content_type == "high" and high_href is None:
                        high_href = candidate_href
                    if candidate_href.lower().endswith(".jpg") and any_jpg_href is None:
                        any_jpg_href = candidate_href
                    if href is None:
                        href = candidate_href
                href = low_href or high_href or any_jpg_href or href
                if not href:
                    logger.warning(
                        f"No graphic href for IOP figure '{caption_id}' in {doi}"
                    )
                    continue

                # Try local file first
                local_path = os.path.join(xml_dir, href)
                saved = FigureExtractor.save_figure_from_local_path(
                    local_path, doi, caption_id, base_path
                )
                if saved:
                    logger.info(
                        f"Saved IOP figure '{caption_id}' for {doi} from local file"
                    )
                    continue

                # Fallback: try IOP CDN
                figure_filename = os.path.basename(href)
                content = None
                if cdn_parts and cache_key:
                    content = self._download_iop_figure_from_content_cdn(
                        issn=issn,
                        volume=volume,
                        issue=issue,
                        elocation_id=elocation_id,
                        figure_filename=figure_filename,
                        cache_key=cache_key,
                    )
                if content:
                    saved = FigureExtractor.save_figure_from_bytes(
                        content, doi, caption_id, base_path
                    )
                    if saved:
                        logger.info(
                            f"Saved IOP figure '{caption_id}' for {doi} from content CDN"
                        )
                else:
                    logger.warning(
                        f"Could not download IOP figure '{caption_id}' for {doi} "
                        "from content CDN; caption saved to info.json only."
                    )
            return has_caption_keyword_match
        except Exception as e:
            logger.warning(f"Error extracting figures from IOP XML for {doi}: {e}")
            return False

    def _extract_paragraphs(self, element):
        paragraphs = element.xpath('.//*[local-name()="p"]')
        other_paragraphs = ""
        comp_paragraphs = ""
        for paragraph in paragraphs:
            paragraph_text = " " + "".join(paragraph.itertext())
            cleaned_text = re.sub(
                r" \.",
                ".",
                re.sub(
                    r" \[[,]*\]| \([,]*\)", " ", re.sub(r"\s+", " ", paragraph_text)
                ),
            )
            if any(
                word in cleaned_text for word in ArticleRelatedKeywords.COMP_KEYWORDS
            ):
                comp_paragraphs += cleaned_text
            else:
                other_paragraphs += cleaned_text
        return other_paragraphs, comp_paragraphs

    def _append_sections_to_df(
        self,
        abstract,
        req_sections,
        doi,
        tables,
        article_title,
        publication_name,
        publisher,
        has_caption_keyword_match: bool = False,
    ):
        def _append_section(section):
            """
            Append the section to the dictionary of all paragraphs
            """
            other_section, comp_section = self._extract_paragraphs(section)
            encoded_other_section = other_section.encode("unicode_escape").decode(
                "utf-8"
            )
            encoded_comp_section = comp_section.encode("unicode_escape").decode("utf-8")
            return encoded_other_section, encoded_comp_section

        all_req_data = {
            "doi": doi,
            "article_title": article_title,
            "publication_name": publication_name,
            "publisher": publisher,
            "abstract": "",
            "introduction": "",
            "exp_methods": "",
            "comp_methods": "",
            "results_discussion": "",
            "conclusion": "",
            "is_property_mentioned": "0",
        }
        abstract_text = ""
        if abstract:
            try:
                abstract_text = "".join(
                    abstract[0].xpath('.//*[local-name()="p"]/text()')
                )
            except:
                abstract_text = ""
            all_req_data["abstract"] = abstract_text
        for section in req_sections:
            title_element = section.xpath('./child::*[local-name()="title"]')
            if title_element:
                title = " ".join([t.text for t in title_element]).lower()
                if any(
                    word in title
                    for word in self.article_related_keywords.get_section_keywords(
                        "introduction"
                    )
                ):
                    other_section, comp_section = _append_section(section)
                    if other_section != "":
                        if "introduction" in all_req_data:
                            all_req_data["introduction"] += other_section
                        else:
                            all_req_data["introduction"] = other_section
                elif any(
                    word in title
                    for word in self.article_related_keywords.get_section_keywords(
                        "methods"
                    )
                ):
                    other_section, comp_section = _append_section(section)
                    if other_section != "":
                        if "exp_methods" in all_req_data:
                            all_req_data["exp_methods"] += other_section
                        else:
                            all_req_data["exp_methods"] = other_section
                    if comp_section != "":
                        if "comp_methods" in all_req_data:
                            all_req_data["comp_methods"] += comp_section
                        else:
                            all_req_data["comp_methods"] = comp_section
                elif any(
                    word in title
                    for word in self.article_related_keywords.get_section_keywords(
                        "results_discussion"
                    )
                ):
                    other_section, comp_section = _append_section(section)
                    whole_section = other_section + comp_section
                    if whole_section != "":
                        if "results_discussion" in all_req_data:
                            all_req_data["results_discussion"] += whole_section
                        else:
                            all_req_data["results_discussion"] = whole_section
                elif any(
                    word in title
                    for word in self.article_related_keywords.get_section_keywords(
                        "conclusion"
                    )
                ):
                    other_section, comp_section = _append_section(section)
                    if other_section != "":
                        if "conclusion" in all_req_data:
                            all_req_data["conclusion"] += other_section
                        else:
                            all_req_data["conclusion"] = other_section

        if tables:
            tables_content = "\n".join(tables)
            all_req_data["results_discussion"] += "\n" + tables_content

        # Check if property is mentioned in the article
        total_text = f"#TITLE:\n{all_req_data['article_title']}\n\n# ABSTRACT:\n{all_req_data["abstract"]}\n\n# INTRODUCTION:\n{all_req_data["introduction"]}\n\n# EXPERIMENTAL SYNTHESIS:\n{all_req_data["exp_methods"]}\n\n# COMPUTATIONAL METHODOLOGY:\n{all_req_data["comp_methods"]}\n\n# RESULTS AND DISCUSSION:\n{all_req_data["results_discussion"]}\n\n# CONCLUSION\n{all_req_data["conclusion"]}"
        has_text_property_match = False
        for item in self.property_keywords.values():
            for keyword in item:
                if keyword in total_text:
                    has_text_property_match = True
                    all_req_data["is_property_mentioned"] = "1"
                    modified_doi = doi.replace("/", "_")
                    if self.vector_db_manager.database_exists(modified_doi):
                        logger.warning(
                            f"Vector Database already exists for {doi}...Skipping..."
                        )
                    else:
                        logger.info(
                            f"Target property is mentioned in {doi}...Creating vector database..."
                        )
                        self.vector_db_manager.create_database(
                            db_name=modified_doi, article_text=total_text
                        )
                    break
            if has_text_property_match:
                break
        if has_caption_keyword_match and not has_text_property_match:
            all_req_data["is_property_mentioned"] = "1"
            logger.info(
                f"Target caption keyword matched for {doi}; keeping article for downstream extraction."
            )
        if all_req_data["is_property_mentioned"] == "0":
            all_req_data["abstract"] = ""
            all_req_data["introduction"] = ""
            all_req_data["exp_methods"] = ""
            all_req_data["comp_methods"] = ""
            all_req_data["results_discussion"] = ""
            all_req_data["conclusion"] = ""

        # Returning dataframe (appended dictionary for one article)
        return pd.DataFrame([all_req_data])

    ###### Table processing functions ######
    def _process_header(self, head_element):
        """
        Function to process and return the header element of a table
        params: head_element: Element containing the header of the table
        """
        if head_element.xpath('.//*[local-name()="sup"]'):
            self._modify_specific_element(head_element, "sup")
        if head_element.xpath('.//*[local-name()="math"]'):
            self._modify_specific_element(head_element, "math")
        return [
            " ".join(
                "".join(th.itertext()).encode("unicode_escape").decode("utf-8").split()
            ).replace("\\n", "")
            for th in head_element.xpath('.//*[local-name()="th"]')
        ]

    def _process_entries(self, entries):
        """
        Function to process and return the entries (body elements) of a table
        params: entries: List of entries in a row of the table
        """
        row_data = []
        for entry in entries:
            if entry.xpath('.//*[local-name()="sup"]'):
                self._modify_specific_element(entry, "sup")
            if entry.xpath('.//*[local-name()="math"]'):
                self._modify_specific_element(entry, "math")
            if entry.xpath('.//*[local-name()="br"]'):
                row_data.append(
                    [text.strip().replace("\\n", "") for text in entry.xpath("text()")]
                )
            else:
                text = " ".join(
                    "".join(entry.itertext())
                    .encode("unicode_escape")
                    .decode("utf-8")
                    .split()
                ).replace("\\n", "")
                row_data.append(text.strip().replace("\\n", ""))
        return row_data

    def _process_rows(self, rows):
        """
        Function to process and return the rows of a table
        params: rows: List of rows in the table
        """
        all_row_data = []
        for row in rows:
            entries = row.xpath('.//*[local-name()="td"]')
            all_row_data.append(self._process_entries(entries))
        return all_row_data

    def _process_tables(self, tables):
        """
        Function to process and return the tables
        params: tables: List of tables
        """

        def _process_caption(caption_element: etree._Element) -> str:
            """Function to process and return the caption element of a table

            Args:
                caption_element: Element containing the caption of the table

            Returns:
                str: Processed caption text
            """
            return (
                " ".join(
                    "".join(caption_element.itertext())
                    .encode("unicode_escape")
                    .decode("utf-8")
                    .split()
                )
                .strip()
                .replace("\\n", "")
            )

        def _get_preceding_caption(table_element: etree._Element) -> str:
            """
            Given a table element, this function returns the text of the immediately preceding caption element, if any.

            Args:
                table_element: The table element

            Returns:
                str: The text of the caption element, if any
            """
            caption_element = table_element.xpath("preceding-sibling::caption[1]")
            if caption_element:
                return _process_caption(caption_element=caption_element[0])
            return None

        header_data = []
        column_number = []
        all_table_data = []
        caption_data = []

        for table in tables:
            caption = _get_preceding_caption(table)
            if caption:
                head_elements = table.xpath('.//*[local-name()="thead"]')
                if head_elements:
                    head_element = head_elements[0]
                    header = self._process_header(head_element)

                    body_elements = table.xpath('.//*[local-name()="tbody"]')
                    if body_elements:
                        body_element = body_elements[0]
                        rows = self._process_rows(
                            body_element.xpath('.//*[local-name()="tr"]')
                        )
                        header_data.append(header)
                        column_number.append(len(header))
                        all_table_data.append(rows)
                        caption_data.append(caption)
        return header_data, column_number, all_table_data, caption_data

    def _generate_tables(
        self, header_data, column_number, all_table_data, caption_data
    ):
        """
        Function to generate markdown tables
        params: header_data: List of headers of tables
                column_number: List of number of columns in each table
                all_table_data: List of data in each table
        """
        tables = []
        if len(all_table_data) == len(header_data) == len(column_number):
            for i in range(len(all_table_data)):
                newline = "\n" if i == 0 else ""
                markdown_table = f"{newline}Table {i+1}.{caption_data[i]}\n|{'|'.join(header_data[i])}|\n|{'|'.join(['---'] * column_number[i])}|\n"
                for row in all_table_data[i]:
                    row = [
                        str(item) if isinstance(item, list) else item for item in row
                    ]
                    markdown_table += "|" + "|".join(row) + "|\n"
                tables.append(markdown_table)
            return tables
        else:
            return "Error: Data mismatch"

    def _process_articles(self):
        try:
            self._load_and_preprocess_data()
            sql_dataframes = []
            csv_dataframes = []

            if self.doi_list is None:
                iterable = self.df.iterrows()
                total = self.df.shape[0]
            else:
                iterable = enumerate(self.doi_list)
                total = len(self.doi_list)

            self.prepare_iop_files.prepare_files()
            all_available_iop_dois = self.prepare_iop_files.get_all_iop_dois()
            for _, item in tqdm(iterable, total=total, colour="#d6adff"):
                try:
                    if self.doi_list is None:
                        row = item
                        doi = row["doi"]
                    else:
                        doi = item
                        self.df = pd.read_csv(self.metadata_csv_filename)
                        matching_rows = self.df[self.df["doi"] == doi]
                        if matching_rows.empty:
                            logger.warning(
                                f"DOI {doi} for IOP article was not found in the metadata CSV."
                            )
                            continue
                        row = matching_rows.iloc[0]
                    if doi in all_available_iop_dois:
                        modified_doi = doi.replace("/", "_")
                        root = self._find_article(
                            f"{self.iop_folderpath}/{modified_doi}.xml"
                        )
                        if root is None:
                            logger.error(
                                f"Error processing IOP article with DOI {doi}. Skipping..."
                            )
                            self._record_failed_article(doi, "xml_parse_failed")
                            continue
                        has_caption_keyword_match = self._extract_and_save_figures(
                            root, doi, self.iop_folderpath
                        )
                        body_element = root.xpath('.//*[local-name()="body"]')
                        body = body_element[0] if body_element else None
                        if body is None:
                            logger.error(f"Body not found for DOI: {doi}. Skipping...")
                            self._record_failed_article(doi, "body_not_found")
                            continue
                        tables = body.xpath('.//*[local-name()="table"]')
                        if tables:
                            header_data, column_number, all_table_data, caption_data = (
                                self._process_tables(tables)
                            )
                            tables = self._generate_tables(
                                header_data, column_number, all_table_data, caption_data
                            )
                        else:
                            tables = []
                        abstract, sections = self._process_xml(root)
                        row = self._append_sections_to_df(
                            abstract,
                            sections,
                            row["doi"],
                            tables,
                            row["article_title"],
                            row["publication_name"],
                            row["metadata_publisher"],
                            has_caption_keyword_match=has_caption_keyword_match,
                        )
                        sql_dataframes.append(row)
                        csv_dataframes.append(row)
                        if row["is_property_mentioned"].iloc[0] == "1":
                            self.valid_property_articles += 1
                        if len(sql_dataframes) == self.sql_batch_size:
                            final_sql_df = pd.concat(sql_dataframes, ignore_index=True)
                            if self.is_sql_db:
                                self.sql_db_manager.write_to_sql_db(
                                    self.paperdata_table_name, final_sql_df
                                )
                            sql_dataframes = []
                            time.sleep(5)
                        if len(csv_dataframes) == self.csv_batch_size:
                            final_csv_df = pd.concat(csv_dataframes, ignore_index=True)
                            self.csv_db_manager.write_to_csv(
                                final_csv_df,
                                self.csv_path,
                                self.keyword,
                                self.source,
                                self.csv_batch_size,
                            )
                            csv_dataframes = []
                            time.sleep(5)

                except KeyboardInterrupt as kie:
                    logger.error(f"Keyboard Interruption Detected. {kie}. Exiting...")
                    raise KeyboardInterruptHandler()
                except Exception as e:
                    logger.error(
                        f"Error processing IOP article with DOI {row["doi"]}: {e}"
                    )
                    continue

                # Append any remaining dataframes at the last
                if sql_dataframes:
                    remaining_sql_df = pd.concat(sql_dataframes, ignore_index=True)
                    if self.is_sql_db:
                        self.sql_db_manager.write_to_sql_db(
                            self.paperdata_table_name, remaining_sql_df
                        )
                if csv_dataframes:
                    remaining_csv_df = pd.concat(csv_dataframes, ignore_index=True)
                    self.csv_db_manager.write_to_csv(
                        remaining_csv_df,
                        self.csv_path,
                        self.keyword,
                        self.source,
                        self.csv_batch_size,
                    )
                time.sleep(0.0001)
        except KeyboardInterrupt:
            logger.error(f"Keyboard Interruption Detected. Exiting...")
            raise KeyboardInterruptHandler()
        except Exception as e:
            logger.error(f"Error processing IOP articles: {e}")
            raise Exception(f"Error processing IOP articles: {e}")

    def process_iop_articles(self):
        """Run IOP article processing workflow"""
        logger.verbose(f"\n\nIOP articles processing started...\n\n")
        self._process_articles()
        logger.verbose(f"\n\nIOP articles processing completed...\n\n")
        logger.info(f"\nTotal valid property articles: {self.valid_property_articles}")
        if self.failed_automated_count > 0:
            logger.warning(
                f"Total failed IOP articles (download/parse): {self.failed_automated_count}"
            )
            if self.save_failed_automated_report:
                logger.info(
                    f"Failed automated report saved to: {self.failed_automated_report_path}"
                )
