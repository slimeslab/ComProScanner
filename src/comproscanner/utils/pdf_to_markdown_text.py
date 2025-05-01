"""
pdf_to_markdown_text.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-03-2025
"""

# Standard library imports
import re
import os
import pandas as pd
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings

# Custom imports
from .error_handler import ValueErrorHandler
from .configs import ArticleRelatedKeywords
from .logger import setup_logger

######## logger Configuration ########
logger = setup_logger("article_processor.log")


class PDFToMarkdownText:
    def __init__(self, source: str = None, num_threads: int = 4):
        """Class to convert PDF to Markdown text.

        Args:
            source (str, required): Source PDF file path or URL. Defaults to None.
            num_threads (int, optional): Number of CPU threads to use for conversion. Defaults to 4.
        """
        self.source = source
        if self.source == None:
            logger.error("Source cannot be empty...")
            raise ValueErrorHandler(f"Source cannot be empty...")
        self.article_keywords = ArticleRelatedKeywords()
        self.num_threads = num_threads
        self.converter = self._setup_converter()

    def _setup_converter(self):
        """Setup document converter with appropriate acceleration options.

        Returns:
            DocumentConverter: Configured document converter
        """
        try:
            if torch.cuda.is_available():
                device = AcceleratorDevice.CUDA
                logger.info("Using CUDA acceleration for PDF processing")
            else:
                device = AcceleratorDevice.CPU
                logger.info("No GPU available, using CPU for PDF processing")

        except ImportError:
            # If torch is not available, fall back to CPU
            device = AcceleratorDevice.CPU
            logger.info("PyTorch not available, using CPU for PDF processing")

        # Configure accelerator options
        accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads, device=device
        )

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # Enable timing profiling for debugging
        settings.debug.profile_pipeline_timings = True

        # Create converter with specified options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        return converter

    def convert_to_markdown(self):
        """Function to convert PDF to Markdown text.

        Returns:
            str: Markdown text.
        """
        try:
            result = self.converter.convert(self.source)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Error converting PDF to Markdown: {e}")

    @staticmethod
    def clean_text(text: str):
        """Function to clean the text.

        Args:
            text (str, required): Text to be cleaned.

        Returns:
            str: Cleaned text.
        """

        def _split_at_references(text: str):
            pattern = r"\n(?:#{1,3})\s*.*?references.*$"
            parts = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            return parts[0] if parts else text

        def _split_into_sections(text: str):
            """
            Function to split the text into sections based on the section headers.

            Args:
                text (str, required): Text to be split into sections.

            Returns:
                list: List of sections.
            """
            pattern = r"\n(?=#{1,3}\s)"
            sections = re.split(pattern, text)
            # Remove empty sections and strip whitespace
            return [section.strip() for section in sections if section.strip()]

        md_text = text.replace("<!-- image -->", "")
        md_text = _split_at_references(md_text)
        sections = _split_into_sections(md_text)

        return sections

    def append_section_to_df(
        self,
        req_sections,
        doi,
        article_title,
        publication_name,
        publisher,
        property_keywords,
        vector_db_manager,
        logger,
    ):
        """
        Function to append the sections to the dataframe.

        Args:
            req_sections (list, required): List of sections.
            doi (str, required): DOI of the article.
            article_title (str, required): Title of the article.
            publication_name (str, required): Name of the publication.
            publisher (str, required): Name of the publisher.
            property_keywords (dict, required): Dict of property keywords.
            logger (logging.Logger, required): Logger object.

        Returns:
            pd.DataFrame: Dataframe containing the article data.
        """
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

        def _get_diff_paragraphs(section: str):
            """
            Function to separate the paragraphs containing the computational keywords from the other paragraphs.
            Args:
                section (str, required): Section text.

            Returns:
                comp_paragraphs (str): Computational paragraphs.
                other_paragraphs (str): Other paragraphs.
            """
            other_paragraphs = ""
            comp_paragraphs = ""
            # split the section into paragraphs based on '\n\n'
            paragraphs = section.split("\n\n")
            for paragraph in paragraphs:
                # check if the paragraph contains any of the keywords
                if any(
                    keyword in paragraph.lower()
                    for keyword in self.article_keywords.COMP_KEYWORDS
                ):
                    comp_paragraphs += paragraph
                else:
                    other_paragraphs += paragraph
            return other_paragraphs, comp_paragraphs

        def _get_folder_names(path="db"):
            # check if the db folder exists
            if not os.path.exists(path):
                return []
            else:
                return [
                    d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
                ]

        def _get_section_type(section_title: str, keywords_dict: dict):
            """
            Function to determine the type of section based on its title and a list of keywords

            Args:
                section_title (str, required): Title of the section.
                keywords_dict (dict, required): Dict of keywords to search for.

            Returns:
                index (int/None): Section index if found, else None
            """
            section_title_lower = section_title.lower()
            for section_type, keywords in keywords_dict.items():
                for keyword in keywords:
                    if any(keyword in word for word in section_title_lower.split()):
                        return section_type
            return None

        def _combine_sections(sections: list, section_keywords: dict):
            final_sections = {
                "abstract": "",
                "introduction": "",
                "experimental_methods": "",
                "computational_methods": "",
                "results_discussion": "",
                "conclusion": "",
            }

            current_section_type = None

            for section in sections:
                # Split section into lines. Get the first line and the rest of the content
                section_lines = section.split("\n")
                first_line = section_lines[0]
                modified_first_line = first_line.lower().replace(" ", "")
                section_content = "\n".join(
                    line.strip() for line in section_lines[1:] if line.strip()
                )

                # Check if this is a new major section
                section_type = _get_section_type(modified_first_line, section_keywords)

                if section_type:
                    current_section_type = section_type

                # Add content to the current section if a valid section type is there
                if current_section_type:
                    other_content, comp_content = _get_diff_paragraphs(section_content)

                    if current_section_type == "methods":
                        # For methods section, split content into experimental and computational
                        if other_content:
                            if final_sections["experimental_methods"]:
                                final_sections["experimental_methods"] += (
                                    "\n" + other_content
                                )
                            else:
                                final_sections["experimental_methods"] = other_content

                        if comp_content:
                            if final_sections["computational_methods"]:
                                final_sections["computational_methods"] += (
                                    "\n" + comp_content
                                )
                            else:
                                final_sections["computational_methods"] = comp_content

                    elif current_section_type == "results_discussion":
                        # For results_discussion, combine both other and computational content
                        content_to_add = other_content
                        if comp_content:
                            if content_to_add:
                                content_to_add += "\n" + comp_content
                            else:
                                content_to_add = comp_content

                        if content_to_add:
                            if final_sections["results_discussion"]:
                                final_sections["results_discussion"] += (
                                    "\n" + content_to_add
                                )
                            else:
                                final_sections["results_discussion"] = content_to_add
                    else:
                        # For other sections, only use other_content
                        if other_content:
                            if final_sections[current_section_type]:
                                final_sections[current_section_type] += (
                                    "\n" + other_content
                                )
                            else:
                                final_sections[current_section_type] = other_content

            return final_sections

        final_sections = _combine_sections(
            req_sections, self.article_keywords.SECTION_TITLE_WORDS
        )
        all_req_data["abstract"] = final_sections["abstract"]
        all_req_data["introduction"] = final_sections["introduction"]
        all_req_data["exp_methods"] = final_sections["experimental_methods"]
        all_req_data["comp_methods"] = final_sections["computational_methods"]
        all_req_data["results_discussion"] = final_sections["results_discussion"]
        all_req_data["conclusion"] = final_sections["conclusion"]

        total_text = f"#TITLE:\n{all_req_data['article_title']}\n\n# ABSTRACT:\n{all_req_data["abstract"]}\n\n# INTRODUCTION:\n{all_req_data["introduction"]}\n\n# EXPERIMENTAL SYNTHESIS:\n{all_req_data["exp_methods"]}\n\n# COMPUTATIONAL METHODOLOGY:\n{all_req_data["comp_methods"]}\n\n# RESULTS AND DISCUSSION:\n{all_req_data["results_discussion"]}\n\n# CONCLUSION\n{all_req_data["conclusion"]}"
        for item in property_keywords.values():
            for keyword in item:
                if keyword in total_text:
                    all_req_data["is_property_mentioned"] = "1"
                    modified_doi = doi.replace("/", "_")
                    created_db_names = _get_folder_names()
                    if modified_doi not in created_db_names:
                        logger.info(
                            f"Target property is mentioned in {doi}...Creating vector database..."
                        )
                        vector_db_manager.create_database(
                            db_name=modified_doi, article_text=total_text
                        )
                        break
                    else:
                        logger.warning(f"Database already exists for {doi}...")
        if all_req_data["is_property_mentioned"] == "0":
            all_req_data["abstract"] = ""
            all_req_data["introduction"] = ""
            all_req_data["exp_methods"] = ""
            all_req_data["comp_methods"] = ""
            all_req_data["results_discussion"] = ""
            all_req_data["conclusion"] = ""

        # Returning dataframe (appended dictionary for one article)
        return pd.DataFrame([all_req_data])
