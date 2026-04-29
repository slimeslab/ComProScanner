import os
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd

# Avoid importing heavy vector DB runtime deps during test module import.
if "langchain_chroma" not in sys.modules:
    _fake_langchain_chroma = types.ModuleType("langchain_chroma")
    _fake_langchain_chroma.Chroma = MagicMock()
    sys.modules["langchain_chroma"] = _fake_langchain_chroma
if "chromadb" not in sys.modules:
    _fake_chromadb = types.ModuleType("chromadb")
    _fake_chromadb.PersistentClient = MagicMock()
    sys.modules["chromadb"] = _fake_chromadb

from comproscanner.comproscanner import ComProScanner


def test_collect_metadata_public_api_calls_fetch_and_filter():
    with (
        patch("comproscanner.comproscanner.FetchMetadata") as mock_fetch_cls,
        patch("comproscanner.comproscanner.FilterMetadata") as mock_filter_cls,
    ):
        scanner = ComProScanner(main_property_keyword="piezoelectric")
        scanner.collect_metadata(
            base_queries=["q1"], extra_queries=["q2"], start_year=2025, end_year=2024
        )

        mock_fetch_cls.assert_called_once_with(
            main_property_keyword="piezoelectric",
            start_year=2025,
            end_year=2024,
            base_queries=["q1"],
            extra_queries=["q2"],
        )
        mock_fetch_cls.return_value.main_fetch.assert_called_once_with()
        mock_filter_cls.assert_called_once_with(main_property_keyword="piezoelectric")
        mock_filter_cls.return_value.update_publisher_information.assert_called_once_with()


def test_process_articles_routes_dois_to_matching_publishers():
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    doi_list = ["10.1/a", "10.2/b", "10.3/c", "10.4/d", "10.5/e"]
    metadata_df = pd.DataFrame(
        {
            "doi": ["10.1/a", "10.2/b", "10.3/c", "10.4/d"],
            "general_publisher": ["elsevier", "springer", "wiley", "iop"],
        }
    )

    mock_elsevier_cls = MagicMock()
    mock_springer_cls = MagicMock()
    mock_wiley_cls = MagicMock()
    mock_iop_cls = MagicMock()

    fake_modules = {
        "comproscanner.article_processors.elsevier_processor": types.SimpleNamespace(
            ElsevierArticleProcessor=mock_elsevier_cls
        ),
        "comproscanner.article_processors.springer_processor": types.SimpleNamespace(
            SpringerArticleProcessor=mock_springer_cls
        ),
        "comproscanner.article_processors.wiley_processor": types.SimpleNamespace(
            WileyArticleProcessor=mock_wiley_cls
        ),
        "comproscanner.article_processors.iop_processor": types.SimpleNamespace(
            IOPArticleProcessor=mock_iop_cls
        ),
    }

    with (
        patch("comproscanner.comproscanner.os.path.exists", return_value=True),
        patch("pandas.read_csv", return_value=metadata_df),
        patch.dict(sys.modules, fake_modules, clear=False),
    ):
        scanner.process_articles(
            property_keywords={"exact_keywords": ["d33"], "substring_keywords": []},
            source_list=["elsevier", "springer", "wiley", "iop"],
            doi_list=doi_list,
        )

    assert mock_elsevier_cls.call_args.kwargs["doi_list"] == ["10.1/a"]
    assert mock_springer_cls.call_args.kwargs["doi_list"] == ["10.2/b"]
    assert mock_wiley_cls.call_args.kwargs["doi_list"] == ["10.3/c"]
    assert mock_iop_cls.call_args.kwargs["doi_list"] == ["10.4/d"]
    mock_elsevier_cls.return_value.process_elsevier_articles.assert_called_once_with()
    mock_springer_cls.return_value.process_springer_articles.assert_called_once_with()
    mock_wiley_cls.return_value.process_wiley_articles.assert_called_once_with()
    mock_iop_cls.return_value.process_iop_articles.assert_called_once_with()


def test_extract_composition_property_data_public_api_smoke_with_no_papers(tmp_path):
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    output_file = tmp_path / "results.json"

    mock_preparator = MagicMock()
    mock_preparator.get_unprocessed_data.return_value = []

    with (
        patch("comproscanner.comproscanner.MatPropDataPreparator", return_value=mock_preparator),
        patch("comproscanner.comproscanner.LLMConfig") as mock_llm_cfg,
        patch("comproscanner.comproscanner.DataCleaner") as mock_cleaner_cls,
    ):
        mock_llm_cfg.return_value.get_llm.return_value = MagicMock()
        mock_cleaner_cls.return_value.get_useful_data.return_value = {}

        scanner.extract_composition_property_data(
            main_extraction_keyword="d33",
            json_results_file=str(output_file),
            checked_doi_list_file=str(tmp_path / "checked.txt"),
        )

    assert os.path.exists(output_file)
    mock_cleaner_cls.return_value.get_useful_data.assert_called_once_with()


def test_process_articles_forwards_pdf_failed_report_args():
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    mock_pdfs_cls = MagicMock()
    fake_modules = {
        "comproscanner.article_processors.pdfs_processor": types.SimpleNamespace(
            PDFsProcessor=mock_pdfs_cls
        )
    }

    with patch.dict(sys.modules, fake_modules, clear=False):
        scanner.process_articles(
            property_keywords={"exact_keywords": ["d33"], "substring_keywords": []},
            source_list=["pdfs"],
            folder_path="/tmp/pdfs",
            save_failed_pdf_report=False,
            failed_pdf_report_path="/tmp/failed_pdf_report.txt",
        )

    assert mock_pdfs_cls.call_args.kwargs["save_failed_pdf_report"] is False
    assert (
        mock_pdfs_cls.call_args.kwargs["failed_pdf_report_path"]
        == "/tmp/failed_pdf_report.txt"
    )
    mock_pdfs_cls.return_value.process_pdfs.assert_called_once_with()


def test_clean_data_public_api_forwards_to_cleaner(tmp_path):
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")

    cleaner = MagicMock()
    cleaner.clean_data_with_relevant_compositions.return_value = {"10.x/test": {}}

    with patch("comproscanner.comproscanner.DataCleaner", return_value=cleaner):
        result = scanner.clean_data(
            json_results_file=str(input_file),
            is_save_separate_results=False,
            is_save_composition_property_file=False,
            cleaning_strategy="full",
        )

    assert result == {"10.x/test": {}}
    cleaner.clean_data_with_relevant_compositions.assert_called_once_with(
        strategy="full"
    )


def test_clean_data_stores_unresolved_compositions(tmp_path):
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    input_file = tmp_path / "input.json"
    input_file.write_text('{"10.x/a": {"composition_data": {"compositions_property_values": {"BaTiO3": 1}}, "synthesis_data": {}, "article_metadata": {}}}', encoding="utf-8")
    unresolved_file = tmp_path / "unresolved.txt"

    cleaner = MagicMock()
    cleaner.clean_data_with_relevant_compositions.return_value = {
        "10.x/a": {"composition_data": {"compositions_property_values": {"BaTiO3": 1}}}
    }
    cleaner.filtered_compositions = {"10.x/a": ["ABBREVIATION", "INVALID"]}
    cleaner.unresolved_compositions = {"10.x/a": ["(BadComp)TiO3", "Na(0.5*x)NbO3"]}
    cleaner.all_data = {"10.x/a": {"composition_data": {"compositions_property_values": {"BaTiO3": 1, "(BadComp)TiO3": 2}}}}

    with patch("comproscanner.comproscanner.DataCleaner", return_value=cleaner):
        scanner.clean_data(
            json_results_file=str(input_file),
            is_save_separate_results=False,
            is_save_composition_property_file=True,
            composition_property_file=str(tmp_path / "comp_prop.json"),
            cleaning_strategy="full",
            is_store_unresolved_compositions=True,
            unresolved_compositions_file=str(unresolved_file),
        )

    assert unresolved_file.exists()
    import json as _json
    report = _json.loads(unresolved_file.read_text(encoding="utf-8"))
    assert "filtered" in report
    assert "unresolved" in report
    assert report["filtered"]["10.x/a"] == ["ABBREVIATION", "INVALID"]
    assert report["unresolved"]["10.x/a"] == ["(BadComp)TiO3", "Na(0.5*x)NbO3"]


def test_evaluate_semantic_public_api_supports_value_error_thresholds():
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    thresholds = {
        (-200, 200): 5,
        (201, 500): 8,
        (-500, -201): 8,
        (501, float("inf")): 10,
        (float("-inf"), -501): 10,
    }

    with patch(
        "comproscanner.comproscanner.MaterialsDataSemanticEvaluator"
    ) as mock_evaluator_cls:
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"ok": True}
        mock_evaluator_cls.return_value = mock_evaluator

        result = scanner.evaluate_semantic(
            ground_truth_file="gt.json",
            test_data_file="test.json",
            extraction_agent_model_name="model-x",
            value_error_thresholds=thresholds,
        )

    assert result == {"ok": True}
    assert mock_evaluator_cls.call_args.kwargs["value_error_thresholds"] == thresholds
    assert (
        mock_evaluator.evaluate.call_args.kwargs["value_error_thresholds"] == thresholds
    )


def test_evaluate_agentic_public_api_supports_value_error_thresholds():
    scanner = ComProScanner(main_property_keyword="piezoelectric")
    thresholds = {
        (-200, 200): 5,
        (201, 500): 8,
        (-500, -201): 8,
        (501, float("inf")): 10,
        (float("-inf"), -501): 10,
    }

    with patch(
        "comproscanner.comproscanner.MaterialsDataAgenticEvaluatorFlow"
    ) as mock_flow_cls:
        mock_flow = MagicMock()
        mock_flow.kickoff.return_value = {"ok": True}
        mock_flow_cls.return_value = mock_flow

        result = scanner.evaluate_agentic(
            ground_truth_file="gt.json",
            test_data_file="test.json",
            extraction_agent_model_name="model-y",
            value_error_thresholds=thresholds,
        )

    assert result == {"ok": True}
    assert mock_flow_cls.call_args.kwargs["value_error_thresholds"] == thresholds
    mock_flow.kickoff.assert_called_once_with()
