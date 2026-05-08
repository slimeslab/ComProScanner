"""
comproscanner.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 02-04-2025
"""

# Standard library imports
import time
import json
import os
from typing import Optional, Tuple, List, Dict, Union, Any

# Third-party imports
from tqdm import tqdm
from crewai import LLM
import numpy as np

# Custom imports
# metadata
from .metadata_extractor.fetch_metadata import FetchMetadata
from .metadata_extractor.filter_metadata import FilterMetadata

# extract_compro_data
from .utils.data_preparator import MatPropDataPreparator
from .extract_flow.main_extraction_flow import DataExtractionFlow
from .utils.get_paper_data import PaperMetadataExtractor
from .utils.save_results import SaveResults
from .post_processing.data_cleaner import (
    CleaningStrategy,
    DataCleaner,
)

# evaluation
from .post_processing.evaluation.semantic_evaluator import (
    MaterialsDataSemanticEvaluator,
)
from .post_processing.evaluation.eval_flow.eval_flow import (
    MaterialsDataAgenticEvaluatorFlow,
)

# utils
from .utils.error_handler import ValueErrorHandler, KeyboardInterruptHandler
from .utils.logger import setup_logger
from .utils.configs.rag_config import RAGConfig
from .utils.configs.llm_config import LLMConfig

# configure logger
logger = setup_logger("comproscanner.log")


class ComProScanner:
    def __init__(self, main_property_keyword: str = None):
        self.main_property_keyword = main_property_keyword
        if self.main_property_keyword is None:
            raise ValueErrorHandler(
                "Please provide a main property keyword to proceed."
            )

    def collect_metadata(
        self,
        base_queries: Optional[list] = None,
        extra_queries: Optional[list] = None,
        start_year: int = int(time.strftime("%Y")),
        end_year: int = int(time.strftime("%Y")) - 2,
    ):
        """Collect metadata from the articles.

        Args:
            base_queries (list, optional): List of base queries to search for in the articles. Defaults to None.
            extra_queries (list, optional): List of extra queries to search for in the articles. Defaults to None.
            start_year (int, optional): Start year for the search. Defaults to int(time.strftime("%Y")).
            end_year (int, optional): End year for the search. Defaults to int(time.strftime("%Y")) - 2.
        """
        if start_year < end_year:
            raise ValueErrorHandler(
                message="Start year should be greater than the end year."
            )
        if start_year > int(time.strftime("%Y")):
            raise ValueErrorHandler(
                message="Start year cannot be greater than the current year."
            )
        if start_year == end_year:
            raise ValueErrorHandler(
                message="Start year and End year cannot be the same."
            )
        # Fetch metadata
        fetch_metadata = FetchMetadata(
            main_property_keyword=self.main_property_keyword,
            start_year=start_year,
            end_year=end_year,
            base_queries=base_queries,
            extra_queries=extra_queries,
        )
        fetch_metadata.main_fetch()

        # Filter metadata
        filter_metadata = FilterMetadata(
            main_property_keyword=self.main_property_keyword
        )
        filter_metadata.update_publisher_information()

    def process_articles(
        self,
        property_keywords: dict = None,
        source_list: list = ["elsevier", "wiley", "iop", "springer"],
        folder_path: str = None,
        sql_batch_size: int = 500,
        csv_batch_size: int = 1,
        start_row: int = None,
        end_row: int = None,
        doi_list: list = None,
        is_sql_db: bool = False,
        is_save_xml: bool = False,
        is_save_pdf: bool = False,
        rag_db_path: str = "db",
        chunk_size: int = 1000,
        chunk_overlap: int = 25,
        embedding_model: str = "huggingface:thellert/physbert_cased",
        caption_keywords: Optional[Dict] = None,
        save_failed_pdf_report: bool = True,
        failed_pdf_report_path: Optional[str] = None,
        save_failed_automated_report: bool = True,
        failed_automated_report_path: Optional[str] = None,
    ):
        """Process articles for the main property keyword.

        Args:
            property_keywords (dict, required): A dictionary of property keywords which will be used for filtering sentences and should look like the following::

                {
                    "exact_keywords": ["example1", "example2"],
                    "substring_keywords": [" example 1 ", " example 2 "],
                }
            source_list (list, optional): List of sources to process the articles from. Defaults to ["elsevier", "wiley", "iop", "springer"] - currently supported publishers.
            folder_path (str, optional): Path to the folder containing PDFs. Defaults to None.
            sql_batch_size (int, optional): The number of rows to write to the database at once (Applicable only if is_sql_db is True). Defaults to 500.
            csv_batch_size (int, optional): The number of rows to write to the CSV file at once. Defaults to 1.
            start_row (int, optional): Start row to process the articles from. Defaults to None.
            end_row (int, optional): End row to process the articles to. Defaults to None.
            doi_list (list, optional): List of DOIs to process the articles for. Defaults to None.
            is_sql_db (bool, optional): A flag to indicate if the data should be written to the database. Defaults to False.
            is_save_xml (bool, optional): A flag to indicate if the XML files should be saved. Defaults to False.
            is_save_pdf (bool, optional): A flag to indicate if the PDF files should be saved. Defaults to False.
            rag_db_path (str, optional): Path to the vector database. Defaults to 'db'.
            chunk_size (int, optional): Size of the chunks to split the input text into. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between the chunks. Defaults to 25.
            embedding_model (str, optional): Name of the embedding model. Defaults to 'thellert/physbert_cased'.
            caption_keywords (dict, optional): Keywords to match figure captions for figure extraction. Same format as property_keywords with "exact_keywords" and "substring_keywords" keys. Defaults to None (no figure extraction).
            save_failed_pdf_report (bool, optional): For `pdfs` source only. If True, save skipped/failed filename-based DOI fallback cases to a text report. Defaults to True.
            failed_pdf_report_path (str, optional): For `pdfs` source only. Custom path for failed PDF filename report. Defaults to None (uses `{folder_path}/failed_pdf_filenames.txt`).
            save_failed_automated_report (bool, optional): For automated publisher sources (elsevier, springer, iop, wiley). If True, save failed/unparseable articles to a report. Defaults to True.
            failed_automated_report_path (str, optional): Custom path for the automated failure report. Defaults to None (uses `results/failed_automated_articles.txt`).

        Raises:
            ValueErrorHandler: If property_keywords is not provided.
        """
        if property_keywords is None:
            raise ValueErrorHandler(
                message="Please provide property_keywords dictionary to proceed."
            )
        source_list = [source.lower() for source in source_list]
        rag_config = RAGConfig(
            rag_db_path=rag_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )

        routed_doi_list = {
            "elsevier": doi_list,
            "springer": doi_list,
            "wiley": doi_list,
            "iop": doi_list,
        }
        publisher_sources = {"elsevier", "springer", "wiley", "iop"}
        selected_publisher_sources = set(source_list).intersection(publisher_sources)

        # If a manual DOI list is provided, route each DOI to the matching publisher.
        if doi_list is not None and len(selected_publisher_sources) > 0:
            for src in publisher_sources:
                routed_doi_list[src] = []
            try:
                import pandas as pd

                metadata_file = (
                    f"results/{self.main_property_keyword}_metadata.csv"
                )
                if not os.path.exists(metadata_file):
                    logger.warning(
                        f"Metadata file '{metadata_file}' not found. "
                        "Using provided DOI list for all selected sources."
                    )
                    for src in publisher_sources:
                        routed_doi_list[src] = doi_list
                else:
                    metadata_df = pd.read_csv(
                        metadata_file, dtype=str, low_memory=False
                    ).fillna("")
                    if (
                        "doi" not in metadata_df.columns
                        or "general_publisher" not in metadata_df.columns
                    ):
                        logger.warning(
                            "Metadata file is missing required columns "
                            "('doi', 'general_publisher'). "
                            "Using provided DOI list for all selected sources."
                        )
                        for src in publisher_sources:
                            routed_doi_list[src] = doi_list
                    else:
                        doi_to_publisher = {
                            str(row["doi"]).strip(): str(
                                row["general_publisher"]
                            ).strip().lower()
                            for _, row in metadata_df.iterrows()
                            if str(row["doi"]).strip()
                        }

                        unresolved_dois = []
                        filtered_out_dois = []
                        for doi in doi_list:
                            doi_key = str(doi).strip()
                            publisher = doi_to_publisher.get(doi_key)
                            if publisher in selected_publisher_sources:
                                routed_doi_list[publisher].append(doi)
                            elif publisher is None or publisher == "":
                                unresolved_dois.append(doi)
                            else:
                                filtered_out_dois.append((doi, publisher))

                        if unresolved_dois:
                            logger.warning(
                                f"{len(unresolved_dois)} DOI(s) were not found in metadata "
                                "and were skipped."
                            )
                        if filtered_out_dois:
                            logger.info(
                                f"{len(filtered_out_dois)} DOI(s) were skipped because their "
                                "publisher is not in source_list."
                            )
            except Exception as e:
                logger.warning(
                    f"Failed to route DOI list by publisher from metadata: {e}. "
                    "Using provided DOI list for all selected sources."
                )
                for src in publisher_sources:
                    routed_doi_list[src] = doi_list

        # Process Elsevier articles
        if "elsevier" in source_list and (
            doi_list is None or len(routed_doi_list["elsevier"]) > 0
        ):
            from .article_processors.elsevier_processor import ElsevierArticleProcessor

            elsevier_processor = ElsevierArticleProcessor(
                main_property_keyword=self.main_property_keyword,
                property_keywords=property_keywords,
                sql_batch_size=sql_batch_size,
                csv_batch_size=csv_batch_size,
                start_row=start_row,
                end_row=end_row,
                doi_list=routed_doi_list["elsevier"],
                is_sql_db=is_sql_db,
                is_save_xml=is_save_xml,
                rag_config=rag_config,
                caption_keywords=caption_keywords,
                save_failed_automated_report=save_failed_automated_report,
                failed_automated_report_path=failed_automated_report_path,
            )
            elsevier_processor.process_elsevier_articles()

        # Process Springer articles
        if "springer" in source_list and (
            doi_list is None or len(routed_doi_list["springer"]) > 0
        ):
            from .article_processors.springer_processor import SpringerArticleProcessor

            springer_processor = SpringerArticleProcessor(
                main_property_keyword=self.main_property_keyword,
                property_keywords=property_keywords,
                sql_batch_size=sql_batch_size,
                csv_batch_size=csv_batch_size,
                start_row=start_row,
                end_row=end_row,
                doi_list=routed_doi_list["springer"],
                is_sql_db=is_sql_db,
                is_save_xml=is_save_xml,
                rag_config=rag_config,
                caption_keywords=caption_keywords,
                save_failed_automated_report=save_failed_automated_report,
                failed_automated_report_path=failed_automated_report_path,
            )
            springer_processor.process_springer_articles()

        # Process Wiley articles
        if "wiley" in source_list and (
            doi_list is None or len(routed_doi_list["wiley"]) > 0
        ):
            from .article_processors.wiley_processor import WileyArticleProcessor

            wiley_processor = WileyArticleProcessor(
                main_property_keyword=self.main_property_keyword,
                property_keywords=property_keywords,
                sql_batch_size=sql_batch_size,
                csv_batch_size=csv_batch_size,
                start_row=start_row,
                end_row=end_row,
                doi_list=routed_doi_list["wiley"],
                is_sql_db=is_sql_db,
                is_save_pdf=is_save_pdf,
                rag_config=rag_config,
                caption_keywords=caption_keywords,
                save_failed_automated_report=save_failed_automated_report,
                failed_automated_report_path=failed_automated_report_path,
            )
            wiley_processor.process_wiley_articles()

        # Process IOP articles
        if "iop" in source_list and (
            doi_list is None or len(routed_doi_list["iop"]) > 0
        ):
            from .article_processors.iop_processor import IOPArticleProcessor

            iop_processor = IOPArticleProcessor(
                main_property_keyword=self.main_property_keyword,
                property_keywords=property_keywords,
                sql_batch_size=sql_batch_size,
                csv_batch_size=csv_batch_size,
                start_row=start_row,
                end_row=end_row,
                doi_list=routed_doi_list["iop"],
                is_sql_db=is_sql_db,
                rag_config=rag_config,
                caption_keywords=caption_keywords,
                save_failed_automated_report=save_failed_automated_report,
                failed_automated_report_path=failed_automated_report_path,
            )
            iop_processor.process_iop_articles()

        # Process PDFs
        if "pdfs" in source_list:
            from .article_processors.pdfs_processor import PDFsProcessor

            pdf_processor = PDFsProcessor(
                folder_path=folder_path,
                main_property_keyword=self.main_property_keyword,
                property_keywords=property_keywords,
                sql_batch_size=sql_batch_size,
                csv_batch_size=csv_batch_size,
                is_sql_db=is_sql_db,
                rag_config=rag_config,
                caption_keywords=caption_keywords,
                save_failed_pdf_report=save_failed_pdf_report,
                failed_pdf_report_path=failed_pdf_report_path,
            )
            pdf_processor.process_pdfs()

    def extract_composition_property_data(
        self,
        main_extraction_keyword: str = None,
        start_row: int = 0,
        num_rows: int = None,
        is_test_data_preparation=False,
        test_doi_list_file=None,
        total_test_data: int = 50,
        is_only_consider_test_doi_list: bool = False,
        test_random_seed: int = 42,
        checked_doi_list_file: str = "checked_dois.txt",
        json_results_file: str = "results.json",
        csv_results_file: str = "results.csv",
        is_extract_synthesis_data: bool = True,
        is_save_csv: bool = False,
        is_save_relevant: bool = True,
        materials_data_identifier_query: str = None,  # Will be set based on the main_property_keyword if not provided
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        output_log_folder: Optional[str] = None,
        is_log_json: bool = False,
        task_output_folder: Optional[str] = None,
        verbose: bool = True,
        temperature: float = 0.1,
        top_p: float = 0.9,
        timeout: int = 60,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        rag_db_path: str = "db",
        embedding_model: str = "huggingface:thellert/physbert_cased",
        rag_chat_model: str = "gpt-4o-mini",
        rag_max_tokens: int = 512,
        rag_top_k: int = 3,
        rag_base_url: Optional[str] = None,
        vlm_model: str = "gemini/gemini-3-flash-preview",
        related_figures_base_path: Optional[str] = None,
        formula_instruction: Optional[str] = None,
        equation_model: Optional[str] = None,
        **flow_optional_args,
    ):
        """Extract the composition-property data and synthesis data if the property is present in the article.

        Args:
            main_extraction_keyword (str, required): The main keyword to extract the composition-property data for.
            start_row (int, optional): Start row to process the articles from. Defaults to 0.
            num_rows (int, optional): Number of rows to process the articles for. Defaults to all rows.
            is_test_data_preparation (bool, optional): A flag to indicate if the test data preparation is required. Defaults to False.
            test_doi_list_file (str, optional): Path to the file containing the test DOIs. Defaults to None.
            total_test_data (int, optional): Total number of test data. Defaults to 50 if not provided and is_test_data_preparation is True.
            is_only_consider_test_doi_list (bool, optional): A flag to indicate if only the test DOI list should be considered. Defaults to False. Should be set to True if the test_doi_list_file has required number of test DOIs.
            test_random_seed (int, optional): Random seed for test data preparation. Defaults to 42.
            checked_doi_list_file (list, optional): List of DOIs which have been checked already. Defaults to "checked_dois.txt".
            json_results_file (str, optional): Path to the JSON results file. Defaults to "results.json".
            csv_results_file (str, optional): Path to the CSV results file. Defaults to "results.csv".
            is_extract_synthesis_data (bool, optional): A flag to indicate if the synthesis data should be extracted. Defaults to True.
            is_save_csv (bool, optional): A flag to indicate if the results should be saved in the CSV file. Defaults to False.
            is_save_relevant (bool, optional): A flag to indicate if only papers with composition-property data should be saved. If True, only saves papers with composition data. If False, saves all processed papers. Defaults to True.
            materials_data_identifier_query (str, optional): Query to identify the materials data. Must be an 'yes/no' answer. Defaults to "Is there any material chemical composition and corresponding {main_property_keyword} value mentioned in the paper? GIVE ONE WORD ANSWER. Either yes or no."
            model (str, optional): The model to use (defaults to "gpt-4o-mini")
            api_base (str, optional): Base URL for standard API endpoints
            base_url (str, optional): Base URL for the model service
            api_key (str, optional): API key for the model service
            output_log_folder (str, optional): Base folder path to save logs. Logs will be saved in {output_log_folder}/{doi}/ subdirectory. Logs will be in JSON format if is_log_json is True, otherwise plain text. Defaults to None.
            task_output_folder (str, optional): Base folder path to save task outputs. Task outputs will be saved as .txt files in {task_output_folder}/{doi}/ subdirectory. Defaults to None.
            is_log_json (bool, optional): Flag to save logs in JSON format. Defaults to False.
            verbose (bool, optional): Flag to enable verbose output inside the terminal (defaults to True)
            temperature (float, optional): Temperature for text generation - controls randomness (defaults to 0.1)
            top_p (float, optional): Nucleus sampling parameter for text generation - controls diversity (defaults to 0.9)
            timeout (int, optional): Request timeout in seconds (defaults to 60)
            frequency_penalty (float, optional): Frequency penalty for text generation
            max_tokens (int, optional): Maximum tokens for completion
            rag_db_path (str, optional): Path to the vector database. Defaults to 'db'.
            embedding_model (str, optional): Name of the embedding model for RAG. Defaults to 'huggingface:thellert/physbert_cased'.
            rag_chat_model (str, optional): Name of the chat model for RAG. Defaults to 'gpt-4o-mini'.
            rag_max_tokens (int, optional): Maximum tokens for completion for RAG. Defaults to 512.
            rag_top_k (int, optional): Top k value for sampling for RAG. Defaults to 3.
            rag_base_url (str, optional): Base URL for the RAG model service.
            vlm_model (str, optional): Vision LLM model for graph data extraction from saved figures. Defaults to "gemini/gemini-3-flash-preview".
            related_figures_base_path (str, optional): Base path where saved figures are stored. Defaults to
                "results/extracted_data/{main_property_keyword}/related_figures".
            formula_instruction (str, optional): Custom instruction for the EquationTool used to derive
                doped chemical formulas. When None the tool's built-in perovskite instruction is used.
            equation_model (str, optional): Explicit litellm model name for EquationTool (for example,
                "openai/gpt-4o-mini", "gemini/gemini-3-flash-preview"). When None, EquationTool auto-selects
                based on available API keys.
            **flow_optional_args (Any): Optional keyword arguments for the MaterialsFlow class.

        Raises:
            ValueErrorHandler: If main_extraction_keyword is not provided.
        """
        if main_extraction_keyword is None:
            logger.error(
                "main_extraction_keyword cannot be None. Please provide a valid keyword. Exiting..."
            )
            raise ValueErrorHandler(
                message="Please provide main_extraction_keyword to proceed for identifying sentences based on property."
            )
        if is_test_data_preparation and test_doi_list_file is None:
            logger.error("Test data file name is required for test data preparation.")
            raise ValueErrorHandler(
                message="Test data file name is required for test data preparation."
            )
        self.is_test_data_preparation = is_test_data_preparation
        if is_test_data_preparation:
            self.test_doi_list_file = test_doi_list_file
            if total_test_data is None:
                self.total_test_data = 50
            else:
                self.total_test_data = total_test_data
        llm_config = LLMConfig(
            model=model,
            api_base=api_base,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )
        llm = llm_config.get_llm()
        rag_config = RAGConfig(
            rag_db_path=rag_db_path,
            embedding_model=embedding_model,
            rag_chat_model=rag_chat_model,
            rag_max_tokens=rag_max_tokens,
            rag_top_k=rag_top_k,
            rag_base_url=rag_base_url,
        )
        if related_figures_base_path is None:
            related_figures_base_path = (
                f"results/extracted_data/{self.main_property_keyword}/related_figures"
            )
        if materials_data_identifier_query is None:
            materials_data_identifier_query = f"Is there any material chemical composition and corresponding {self.main_property_keyword} value mentioned in the paper? Give one word answer. Either yes or no."
        preparator = MatPropDataPreparator(
            main_property_keyword=self.main_property_keyword,
            main_extraction_keyword=main_extraction_keyword,
            json_results_file=json_results_file,
            start_row=start_row,
            num_rows=num_rows,
            is_test_data_preparation=is_test_data_preparation,
            test_doi_list_file=test_doi_list_file,
            is_only_consider_test_doi_list=is_only_consider_test_doi_list,
            total_test_data=total_test_data,
            test_random_seed=test_random_seed,
            checked_doi_list_file=checked_doi_list_file,
        )
        paper_data_list = preparator.get_unprocessed_data()

        paper_metadata_extractor = PaperMetadataExtractor()

        def _has_composition_data(comp_data):
            """Check if composition data contains actual values"""
            if not comp_data or not isinstance(comp_data, dict):
                return False

            comp_values = comp_data.get("compositions_property_values", {})
            return bool(
                comp_values and isinstance(comp_values, dict) and len(comp_values) > 0
            )

        test_dois_with_data = []
        if is_test_data_preparation:
            if os.path.exists(test_doi_list_file):
                try:
                    with open(test_doi_list_file, "r") as f:
                        test_dois_with_data = [
                            line.strip() for line in f.readlines() if line.strip()
                        ]
                        if test_dois_with_data[-1] == "":
                            test_dois_with_data.pop()
                except Exception as e:
                    logger.warning(
                        f"Error reading test DOI file {test_doi_list_file}: {str(e)}"
                    )

        # Extract composition-property and synthesis data
        for paper_data in tqdm(
            paper_data_list,
            total=len(paper_data_list),
            desc="Processing Papers",
            unit="paper",
        ):
            logger.debug(f"\n\nProcessing DOI: {paper_data['doi']}")
            current_doi = paper_data["doi"]

            try:
                try:
                    if paper_data["comp_prop_text"].strip() == "":
                        logger.warning(
                            f"No composition-property text data for DOI: {paper_data['doi']}. Skipping..."
                        )
                        # add the DOI to checked DOIs file
                        try:
                            dir_path = os.path.dirname(checked_doi_list_file)
                            if dir_path:
                                os.makedirs(dir_path, exist_ok=True)
                            with open(checked_doi_list_file, "a") as f:
                                logger.info(
                                    f"Adding DOI to checked list: {paper_data['doi']}"
                                )
                                f.write(f"{paper_data['doi']}\n")
                        except Exception as e:
                            logger.error(
                                f"Error writing to checked DOIs file {checked_doi_list_file}: {str(e)}"
                            )
                        continue
                    flow = DataExtractionFlow(
                        doi=paper_data["doi"],
                        main_extraction_keyword=main_extraction_keyword,
                        composition_property_text_data=paper_data["comp_prop_text"],
                        synthesis_text_data=paper_data["synthesis_text"],
                        llm=llm,
                        materials_data_identifier_query=materials_data_identifier_query,
                        is_extract_synthesis_data=is_extract_synthesis_data,
                        vlm_model=vlm_model,
                        related_figures_base_path=related_figures_base_path,
                        formula_instruction=formula_instruction,
                        equation_model=equation_model,
                        rag_config=rag_config,
                        output_log_folder=output_log_folder,
                        task_output_folder=task_output_folder,
                        is_log_json=is_log_json,
                        verbose=verbose,
                        **flow_optional_args,
                    )
                    result_dict = flow.kickoff()

                    # Extract composition and synthesis data
                    composition_data = result_dict["composition_data"]
                    synthesis_data = result_dict["synthesis_data"]
                except Exception as e:
                    logger.error(
                        f"Error in MaterialsFlow processing for DOI: {paper_data['doi']}. {e}"
                    )
                    continue

                # Try to get paper metadata
                try:
                    paper_metadata = paper_metadata_extractor.get_article_metadata(
                        paper_data["doi"]
                    )
                    result_dict["article_metadata"] = paper_metadata
                except Exception as e:
                    logger.error(
                        f"Error fetching paper metadata for DOI: {paper_data['doi']}. {e}"
                    )
                    continue

                # Determine if the paper should be saved or not
                should_save = True
                if not _has_composition_data(composition_data):
                    if is_test_data_preparation:
                        should_save = False
                        logger.debug(
                            f"Skipping save for DOI {current_doi} - no composition data (test data preparation mode)"
                        )
                    elif is_save_relevant:
                        should_save = False
                        logger.debug(
                            f"Skipping save for DOI {current_doi} - no composition data (is_save_relevant is True)"
                        )

                # If the paper is relevant or is_save_relevant is False, save the results
                if should_save:
                    try:
                        result_saver = SaveResults(json_results_file, csv_results_file)
                        result_saver.update_in_json(paper_data["doi"], result_dict)
                        if is_save_csv:
                            result_saver.update_in_csv(result_dict)
                    except Exception as e:
                        logger.error(
                            f"Error saving results for DOI: {paper_data['doi']}. {e}"
                        )
                        continue

                # For test data preparation, track DOIs with non-empty composition data
                if is_test_data_preparation and composition_data != {}:
                    if current_doi not in test_dois_with_data:
                        test_dois_with_data.append(current_doi)
                        with open(test_doi_list_file, "w") as f:
                            for doi in test_dois_with_data:
                                f.write(f"{doi}\n")
                        logger.info(
                            f"Added DOI to test list: {current_doi} (now have {len(test_dois_with_data)})"
                        )
                        if len(test_dois_with_data) >= total_test_data:
                            logger.info(
                                f"Reached target test data count: {total_test_data}"
                            )
                            break

                # log the processed DOI in the checked DOIs file
                try:
                    dir_path = os.path.dirname(checked_doi_list_file)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    # Read existing DOIs first
                    existing_dois = set()
                    if os.path.exists(checked_doi_list_file):
                        with open(checked_doi_list_file, "r") as f:
                            existing_dois = set(line.strip() for line in f)

                    # Append only if DOI is not already present
                    if paper_data["doi"] not in existing_dois:
                        with open(checked_doi_list_file, "a") as f:
                            logger.info(
                                f"Adding DOI to checked list: {paper_data['doi']}"
                            )
                            f.write(f"{paper_data['doi']}\n")
                    else:
                        logger.debug(
                            f"DOI already in checked list: {paper_data['doi']}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error writing to checked DOIs file {checked_doi_list_file}: {str(e)}"
                    )

                # Delay before next paper
                time.sleep(5)  # 5-second delay

            except KeyboardInterrupt as kie:
                logger.error(
                    f"Keyboard Interruption Detected. Exiting the program... {kie}"
                )
                raise KeyboardInterruptHandler()
            except Exception as e:
                logger.error(f"Error processing DOI: {paper_data['doi']}. {e}")
                continue

        json_dir = os.path.dirname(json_results_file)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        if not os.path.exists(json_results_file):
            with open(json_results_file, "w", encoding="utf-8") as file:
                json.dump({}, file, indent=2, default=str)

        data_cleaner = DataCleaner(results_file=json_results_file)
        final_data = data_cleaner.get_useful_data()
        # Save back to the JSON file
        with open(json_results_file, "w", encoding="utf-8") as file:
            json.dump(final_data, file, indent=2, default=str)

    def clean_data(
        self,
        json_results_file: str = None,
        is_save_separate_results: bool = True,
        cleaned_json_results_file: str = "cleaned_results.json",
        is_save_composition_property_file: bool = True,
        composition_property_file: str = "composition_property.json",
        cleaning_strategy: str = "full",
        apply_advanced_cleaning: bool = True,
        is_store_unresolved_compositions: bool = False,
        unresolved_compositions_file: str = "unresolved_compositions.json",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Removes extra information (key-value pairs) provided by extracted agents. Finally, cleans the composition-property data based on periodic elements, abbreviations and resolves arithmetic calculations, fractions etc.

        Args:
            json_results_file (str, required): Path to the JSON results file.
            is_save_separate_results (bool, optional): Whether to save separate results file after cleaning. Defaults to True.
            cleaned_json_results_file (str, optional): Path to the cleaned JSON results file with articles having relevant composition-property data. Defaults to "cleaned_results.json".
            is_save_composition_property_file (bool, optional): Whether to save composition-property values to a separate file. Defaults to True.
            composition_property_file (str, optional): Path to the composition-property file containing a dictionary of composition-property data. Defaults to "composition_property.json".
            cleaning_strategy (str, optional): The cleaning strategy to use. Defaults to "full" (with periodic element validation). "basic" (without periodic element validation) is the other option.
            apply_advanced_cleaning (bool, optional): Whether to apply advanced composition cleaning transformations (Miller indices removal, coefficient expansion, normalization, zero-coefficient removal). Defaults to True.
            is_store_unresolved_compositions (bool, optional): Whether to log resolution statistics and save unresolved composition keys to a file. Requires is_save_composition_property_file=True. Defaults to False.
            unresolved_compositions_file (str, optional): Path to the file where unresolved composition keys will be saved. Used only when is_store_unresolved_compositions=True. Defaults to "unresolved_compositions.json".

        Returns:
            tuple: A tuple containing:
                - Dict[str, Any]: Cleaned data based on selected strategy with relevant composition-property data.
                - Dict[str, Any]: All composition-property values collected from the cleaned data. (Returned only if is_save_composition_property_file is True)
        """
        if json_results_file is None:
            logger.error(
                "json_results_file cannot be None. Please provide a valid file path. Exiting..."
            )
            raise ValueErrorHandler(
                message="Please provide json_results_file to proceed for data cleaning."
            )
        if os.path.exists(json_results_file) is False:
            logger.error(
                f"JSON results file {json_results_file} does not exist. Cannot proceed with data cleaning."
            )
            raise ValueErrorHandler(
                message=f"JSON results file {json_results_file} does not exist. Cannot proceed with data cleaning."
            )
        if cleaning_strategy not in [CleaningStrategy.FULL, CleaningStrategy.BASIC]:
            logger.error(
                f"Invalid cleaning strategy: {cleaning_strategy}. Please choose either 'full' or 'basic'."
            )
            raise ValueErrorHandler(
                message=f"Invalid cleaning strategy: {cleaning_strategy}. Please choose either 'full' or 'basic'."
            )
        data_cleaner = DataCleaner(results_file=json_results_file)
        if is_store_unresolved_compositions and is_save_composition_property_file:
            source_composition_count = sum(
                len(article_data.get("composition_data", {}).get("compositions_property_values", {}))
                for article_data in data_cleaner.all_data.values()
                if isinstance(article_data, dict)
            )
        final_data = data_cleaner.clean_data_with_relevant_compositions(
            strategy=cleaning_strategy,
            apply_advanced_cleaning=apply_advanced_cleaning,
        )
        # Save the cleaned data back to the cleaned JSON file
        if is_save_separate_results:
            result_file = cleaned_json_results_file
        else:
            result_file = json_results_file
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(final_data, file, indent=2, default=str)
        logger.info(
            f"Cleaned data saved to {result_file} with relevant composition-property data."
        )
        # Collect all composition-property values from all DOIs
        if is_save_composition_property_file:
            all_composition_property_values = {}
            for _, article_data in final_data.items():
                if "composition_data" in article_data:
                    compositions_property_values = article_data["composition_data"].get(
                        "compositions_property_values", {}
                    )
                    all_composition_property_values.update(compositions_property_values)
            if is_store_unresolved_compositions:
                resolved_count = len(all_composition_property_values)
                filtered_count = sum(len(v) for v in data_cleaner.filtered_compositions.values())
                unresolved_count = sum(len(v) for v in data_cleaner.unresolved_compositions.values())
                logger.info(
                    f"Composition-property pairs — source: {source_composition_count}, "
                    f"filtered: {filtered_count}, unresolved: {unresolved_count}, "
                    f"resolved: {resolved_count}"
                )
                if data_cleaner.filtered_compositions or data_cleaner.unresolved_compositions:
                    dropped_report = {
                        "filtered": data_cleaner.filtered_compositions,
                        "unresolved": data_cleaner.unresolved_compositions,
                    }
                    with open(unresolved_compositions_file, "w", encoding="utf-8") as file:
                        json.dump(dropped_report, file, indent=2)
                    logger.info(
                        f"Filtered and unresolved compositions saved to {unresolved_compositions_file}."
                    )
            # Save all composition-property values to a separate JSON file
            with open(composition_property_file, "w", encoding="utf-8") as file:
                json.dump(all_composition_property_values, file, indent=2, default=str)
            logger.info(
                f"All composition-property values saved to {composition_property_file}."
            )
            return final_data, all_composition_property_values
        return final_data

    def evaluate_semantic(
        self,
        ground_truth_file: str = None,
        test_data_file: str = None,
        extraction_agent_model_name: str = None,
        weights: dict[str, float] = None,
        output_file: str = "semantic_evaluation_result.json",
        is_synthesis_evaluation: bool = True,
        use_semantic_model=True,
        primary_model_name="thellert/physbert_cased",
        fallback_model_name="all-mpnet-base-v2",
        similarity_thresholds=None,
        value_error_thresholds=None,
    ):
        """Evaluate the extracted data using semantic evaluation.

        Args:
            ground_truth_file (str, required): Path to the ground truth file
            test_data_file (str, required): Path to the test data file.
            extraction_agent_model_name (str, required): Name of the agent model used for extraction.
            weights (dict, optional): Weights for evaluation metrics. If not provided, uses default weights: {"compositions_property_values": 0.3, "property_unit": 0.1, "family": 0.1, "method": 0.1, "precursors": 0.15, "characterization_techniques": 0.15, "steps": 0.1}
            output_file (str, optional): Path to the output file for saving the evaluation results. Defaults to "semantic_evaluation_result.json".
            is_synthesis_evaluation (bool, optional): A flag to indicate if synthesis evaluation is required. Defaults to True.
            use_semantic_model (bool, optional): A flag to indicate if semantic model should be used for evaluation. Defaults to True. If False, it will use the fallback SequenceMatcher class from difflib library.
            primary_model_name (str, optional): Name of the primary model for semantic evaluation. Defaults to "thellert/physbert_cased".
            fallback_model_name (str, optional): Name of the fallback model for semantic evaluation. Defaults to "all-mpnet-base-v2".
            similarity_thresholds (dict, optional): Similarity thresholds for evaluation. Defaults to 0.8 for each metric.
            value_error_thresholds (dict, optional): Mapping of ``(min, max)`` tuples to
                absolute error tolerances for numeric property-value comparisons.

        Returns:
            results (dict): Evaluation results containing various metrics.
        """
        if not ground_truth_file:
            raise ValueErrorHandler(
                message="Ground truth file path is required for semantic evaluation."
            )
        if not test_data_file:
            raise ValueErrorHandler(
                message="Test data file path is required for semantic evaluation."
            )
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=use_semantic_model,
            primary_model_name=primary_model_name,
            fallback_model_name=fallback_model_name,
            similarity_thresholds=similarity_thresholds,
            value_error_thresholds=value_error_thresholds,
        )
        results = evaluator.evaluate(
            ground_truth_file=ground_truth_file,
            test_data_file=test_data_file,
            weights=weights,
            output_file=output_file,
            extraction_agent_model_name=extraction_agent_model_name,
            is_synthesis_evaluation=is_synthesis_evaluation,
            value_error_thresholds=value_error_thresholds,
        )
        return results

    def evaluate_agentic(
        self,
        ground_truth_file: str = None,
        test_data_file: str = None,
        extraction_agent_model_name: str = None,
        weights: dict[str, float] = None,
        output_file: str = "agentic_evaluation_result.json",
        is_synthesis_evaluation: bool = True,
        llm: Optional[LLM] = None,
        value_error_thresholds=None,
    ):
        """Evaluate the extracted data using agentic evaluation.

        Args:
            ground_truth_file (str, required): Path to the ground truth file.
            test_data_file (str, required): Path to the test data file.
            extraction_agent_model_name (str, required): Name of the agent model for evaluation.
            weights (dict, optional): Weights for the evaluation metrics. Defaults to None.
            output_file (str, optional): Path to the output file for saving the evaluation results. Defaults to "agentic_evaluation_result.json".
            is_synthesis_evaluation (bool, optional): A flag to indicate if synthesis evaluation is required. Defaults to True.
            llm (LLM, optional): An instance of the LLM class. Defaults to instance of LLM with model="o3-mini"
            value_error_thresholds (dict, optional): Mapping of ``(min, max)`` tuples to
                absolute error tolerances for numeric property-value comparisons.

        Returns:
            results (dict): Evaluation results containing various metrics.
        """
        if not ground_truth_file:
            raise ValueErrorHandler(
                message="Ground truth file path is required for agentic evaluation."
            )
        if not test_data_file:
            raise ValueErrorHandler(
                message="Test data file path is required for agentic evaluation."
            )

        evaluator = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=ground_truth_file,
            test_data_file=test_data_file,
            output_file=output_file,
            extraction_agent_model_name=extraction_agent_model_name,
            is_synthesis_evaluation=is_synthesis_evaluation,
            weights=weights,
            llm=llm,
            value_error_thresholds=value_error_thresholds,
        )
        results = evaluator.kickoff()
        return results
