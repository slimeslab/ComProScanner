"""
test_pdf_to_markdown_text.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-03-2025
"""

import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock, mock_open
from docling.datamodel.base_models import InputFormat
import os

from comproscanner.utils.pdf_to_markdown_text import PDFToMarkdownText
from comproscanner.utils.error_handler import ValueErrorHandler


class TestPDFToMarkdownText:
    """Test cases for PDFToMarkdownText class"""

    def test_initialization_with_valid_source(self):
        """Test initialization with valid source"""
        with patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter"):
            pdf_converter = PDFToMarkdownText(source="test.pdf")
            assert pdf_converter.source == "test.pdf"
            assert pdf_converter.num_threads == 4  # Default value

    def test_initialization_with_custom_threads(self):
        """Test initialization with custom number of threads"""
        with patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter"):
            pdf_converter = PDFToMarkdownText(source="test.pdf", num_threads=8)
            assert pdf_converter.source == "test.pdf"
            assert pdf_converter.num_threads == 8

    def test_initialization_with_invalid_source(self):
        """Test initialization with invalid source"""
        with patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter"):
            with pytest.raises(ValueErrorHandler) as exc_info:
                PDFToMarkdownText(source=None)
            assert "Source cannot be empty" in str(exc_info.value)

    @patch("torch.cuda.is_available")
    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    def test_setup_converter_with_cuda(self, mock_converter, mock_cuda_available):
        """Test _setup_converter method with CUDA available"""
        mock_cuda_available.return_value = True

        with patch("torch.backends.mps.is_available", return_value=False):
            pdf_converter = PDFToMarkdownText(source="test.pdf")
            mock_converter.assert_called_once()
            call_args = mock_converter.call_args[1]
            format_options = call_args.get("format_options", {})
            assert InputFormat.PDF in format_options
            assert (
                format_options[
                    InputFormat.PDF
                ].pipeline_options.accelerator_options.device.name
                == "CUDA"
            )

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    def test_setup_converter_with_cpu(
        self, mock_converter, mock_mps_available, mock_cuda_available
    ):
        """Test _setup_converter method with only CPU available"""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False

        pdf_converter = PDFToMarkdownText(source="test.pdf")
        mock_converter.assert_called_once()
        call_args = mock_converter.call_args[1]
        format_options = call_args.get("format_options", {})
        assert InputFormat.PDF in format_options
        assert (
            format_options[
                InputFormat.PDF
            ].pipeline_options.accelerator_options.device.name
            == "CPU"
        )

    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    def test_convert_to_markdown_success(self, mock_converter):
        """Test convert_to_markdown method with successful conversion"""
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_document.export_to_markdown.return_value = "# Markdown Content"
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result

        pdf_converter = PDFToMarkdownText(source="test.pdf")
        result = pdf_converter.convert_to_markdown()
        assert result == "# Markdown Content"
        mock_converter_instance.convert.assert_called_once_with("test.pdf")
        mock_document.export_to_markdown.assert_called_once()

    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    def test_convert_to_markdown_exception(self, mock_converter):
        """Test convert_to_markdown method with exception"""
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance
        mock_converter_instance.convert.side_effect = Exception("Conversion error")
        pdf_converter = PDFToMarkdownText(source="test.pdf")
        result = pdf_converter.convert_to_markdown()
        assert result is None
        mock_converter_instance.convert.assert_called_once_with("test.pdf")

    def test_clean_text_with_image_placeholders(self):
        """Test clean_text method with image placeholders"""
        markdown_text = """# Title

                            ## Abstract
                            This is the abstract.
                            <!-- image -->

                            ## Introduction
                            This is the introduction.
                            <!-- image -->
                            """
        with patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter"):
            pdf_converter = PDFToMarkdownText(source="test.pdf")
            sections = pdf_converter.clean_text(markdown_text)
            combined_text = "".join(sections)
            assert "<!-- image -->" not in combined_text
            assert "Abstract" in combined_text
            assert "Introduction" in combined_text

    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_append_section_to_df(self, mock_listdir, mock_exists, mock_converter):
        """Test append_section_to_df method"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["folder1", "folder2"]
        sections = [
            "# Title\nSample Title",
            "## Abstract\nThis is the abstract.",
            "## Introduction\nThis is the introduction.",
            "## Methods\nThese are the methods.",
            "## Results\nThese are the results.",
            "## Conclusion\nThis is the conclusion.",
        ]
        mock_vector_db_manager = MagicMock()
        mock_logger = MagicMock()
        property_keywords = {
            "piezoelectric": ["piezoelectric", "piezo"],
            "ferroelectric": ["ferroelectric", "ferro"],
        }
        pdf_converter = PDFToMarkdownText(source="test.pdf")
        result_df = pdf_converter.append_section_to_df(
            req_sections=sections,
            doi="10.1000/test.doi",
            article_title="Test Article",
            publication_name="Test Journal",
            publisher="Test Publisher",
            property_keywords=property_keywords,
            vector_db_manager=mock_vector_db_manager,
            logger=mock_logger,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]["doi"] == "10.1000/test.doi"
        assert result_df.iloc[0]["article_title"] == "Test Article"
        assert result_df.iloc[0]["publication_name"] == "Test Journal"
        assert result_df.iloc[0]["publisher"] == "Test Publisher"

    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_append_section_to_df_with_property_mentioned(
        self, mock_listdir, mock_exists, mock_converter
    ):
        """Test append_section_to_df method when property is mentioned in the text"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["folder1", "folder2"]
        sections = [
            "# Title\nSample Title",
            "## Abstract\nThis is the abstract with piezoelectric properties.",
            "## Introduction\nThis is the introduction.",
            "## Methods\nThese are the methods.",
            "## Results\nThese are the results.",
            "## Conclusion\nThis is the conclusion.",
        ]
        mock_vector_db_manager = MagicMock()
        mock_logger = MagicMock()
        property_keywords = {
            "piezoelectric": ["piezoelectric", "piezo"],
            "ferroelectric": ["ferroelectric", "ferro"],
        }
        pdf_converter = PDFToMarkdownText(source="test.pdf")
        result_df = pdf_converter.append_section_to_df(
            req_sections=sections,
            doi="10.1000/test.doi",
            article_title="Test Article",
            publication_name="Test Journal",
            publisher="Test Publisher",
            property_keywords=property_keywords,
            vector_db_manager=mock_vector_db_manager,
            logger=mock_logger,
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]["doi"] == "10.1000/test.doi"
        assert result_df.iloc[0]["article_title"] == "Test Article"
        assert result_df.iloc[0]["is_property_mentioned"] == "1"
        mock_vector_db_manager.create_database.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch("comproscanner.utils.pdf_to_markdown_text.DocumentConverter")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_append_section_to_df_without_property_mentioned(
        self, mock_listdir, mock_exists, mock_converter
    ):
        """Test append_section_to_df method when property is not mentioned in the text"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["folder1", "folder2"]
        sections = [
            "# Title\nSample Title",
            "## Abstract\nThis is the abstract without any relevant properties.",
            "## Introduction\nThis is the introduction.",
            "## Methods\nThese are the methods.",
            "## Results\nThese are the results.",
            "## Conclusion\nThis is the conclusion.",
        ]

        mock_vector_db_manager = MagicMock()
        mock_logger = MagicMock()

        property_keywords = {
            "piezoelectric": ["piezoelectric", "piezo"],
            "ferroelectric": ["ferroelectric", "ferro"],
        }
        pdf_converter = PDFToMarkdownText(source="test.pdf")
        result_df = pdf_converter.append_section_to_df(
            req_sections=sections,
            doi="10.1000/test.doi",
            article_title="Test Article",
            publication_name="Test Journal",
            publisher="Test Publisher",
            property_keywords=property_keywords,
            vector_db_manager=mock_vector_db_manager,
            logger=mock_logger,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]["doi"] == "10.1000/test.doi"
        assert result_df.iloc[0]["article_title"] == "Test Article"
        assert result_df.iloc[0]["is_property_mentioned"] == "0"
        assert result_df.iloc[0]["abstract"] == ""
        assert result_df.iloc[0]["introduction"] == ""
        assert result_df.iloc[0]["exp_methods"] == ""
        assert result_df.iloc[0]["comp_methods"] == ""
        assert result_df.iloc[0]["results_discussion"] == ""
        assert result_df.iloc[0]["conclusion"] == ""
        mock_vector_db_manager.create_database.assert_not_called()
