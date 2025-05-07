"""
ComProScanner - A package for extracting composition-property data from scientific articles.

This package provides tools to collect metadata, process articles from various sources,
extract composition-property relationships, evaluate extraction performance and visualize data distribution.

Main functions:
- collect_metadata: Collect and filter metadata from scientific articles
- process_articles: Process articles from various sources (Elsevier, Wiley, etc.)
- extract_composition_property_data: Extract composition-property relationships from articles
- evaluate_semantic: Evaluate extraction quality using semantic similarity
- evaluate_agentic: Evaluate extraction quality using agent-based methods
"""

# Import the main class
from .comproscanner import ComProScanner

# Import core configuration classes
from .utils.configs.rag_config import RAGConfig
from .utils.configs.llm_config import LLMConfig

# Import visualisation module
from . import eval_visualiser

# Package version
__version__ = "0.1.0"

# Importing options for "from comproscanner import *"
__all__ = [
    "ComProScanner",
    "collect_metadata",
    "process_articles",
    "extract_composition_property_data",
    "evaluate_semantic",
    "evaluate_agentic",
    "RAGConfig",
    "LLMConfig",
    "eval_visualiser",
    "data_visualiser",
]


def collect_metadata(
    main_property_keyword,
    base_queries=None,
    extra_queries=None,
    start_year=None,
    end_year=None,
):
    """
    Collect metadata from scientific articles.

    Args:
        main_property_keyword (str): The main property keyword to search for
        base_queries (list, optional): List of base queries to search for
        extra_queries (list, optional): List of extra queries to search for
        start_year (int, optional): Start year for the search
        end_year (int, optional): End year for the search
    """
    scanner = ComProScanner(main_property_keyword=main_property_keyword)
    return scanner.collect_metadata(
        base_queries=base_queries,
        extra_queries=extra_queries,
        start_year=start_year,
        end_year=end_year,
    )


def process_articles(
    main_property_keyword,
    property_keywords=None,
    source_list=["elsevier", "wiley", "iop", "springer", "pdfs"],
    **kwargs,
):
    """
    Process articles for the main property keyword.

    Args:
        main_property_keyword (str): The main property keyword to search for
        property_keywords (dict): Dictionary of property keywords for filtering
        source_list (list, optional): List of sources to process
        **kwargs: Additional arguments to pass to the process_articles method
    """
    scanner = ComProScanner(main_property_keyword=main_property_keyword)
    return scanner.process_articles(
        property_keywords=property_keywords, source_list=source_list, **kwargs
    )


def extract_composition_property_data(
    main_property_keyword, main_extraction_keyword=None, **kwargs
):
    """
    Extract composition-property data from articles.

    Args:
        main_property_keyword (str): The main property keyword
        main_extraction_keyword (str): The main keyword to extract data for
        **kwargs: Additional arguments to pass to the extract_composition_property_data method
    """
    scanner = ComProScanner(main_property_keyword=main_property_keyword)
    return scanner.extract_composition_property_data(
        main_extraction_keyword=main_extraction_keyword, **kwargs
    )


def evaluate_semantic(
    ground_truth_file=None,
    test_data_file=None,
    weights=None,
    output_file="semantic_evaluation_result.json",
    extraction_agent_model_name="gpt-4o-mini",
    is_synthesis_evaluation=True,
    use_semantic_model=True,
    primary_model_name="thellert/physbert_cased",
    fallback_model_name="all-MiniLM-L6-v2",
    similarity_thresholds=None,
):
    """
    Evaluate the extracted data using semantic evaluation.

    Args:
        ground_truth_file (str, optional): Path to the ground truth file. Defaults to None.
        test_data_file (str, optional): Path to the test data file. Defaults to None.
        weights (dict, optional): Weights for the evaluation metrics. Defaults to None.
        output_file (str, optional): Path to the output file for saving the evaluation results. Defaults to "semantic_evaluation_result.json".
        extraction_agent_model_name (str, optional): Name of the agent model used for extraction. Defaults to "GPT-4o-mini".
        is_synthesis_evaluation (bool, optional): A flag to indicate if synthesis evaluation is required. Defaults to True.
        use_semantic_model (bool, optional): A flag to indicate if semantic model should be used for evaluation. Defaults to True.
        primary_model_name (str, optional): Name of the primary model for semantic evaluation. Defaults to "thellert/physbert_cased".
        fallback_model_name (str, optional): Name of the fallback model for semantic evaluation. Defaults to "all-MiniLM-L6-v2".
        similarity_thresholds (dict, optional): Similarity thresholds for evaluation. Defaults to 0.8 for each metric.
    """
    scanner = ComProScanner(main_property_keyword="placeholder")
    return scanner.evaluate_semantic(
        ground_truth_file=ground_truth_file,
        test_data_file=test_data_file,
        weights=weights,
        output_file=output_file,
        agent_model_name=extraction_agent_model_name,
        is_synthesis_evaluation=is_synthesis_evaluation,
        use_semantic_model=use_semantic_model,
        primary_model_name=primary_model_name,
        fallback_model_name=fallback_model_name,
        similarity_thresholds=similarity_thresholds,
    )


def evaluate_agentic(
    ground_truth_file=None,
    test_data_file=None,
    output_file="detailed_evaluation.json",
    extraction_agent_model_name="gpt-4o-mini",
    is_synthesis_evaluation=True,
    weights=None,
    llm=None,
):
    """
    Evaluate the extracted data using agentic evaluation.

    Args:
        ground_truth_file (str, optional): Path to the ground truth file. Defaults to None.
        test_data_file (str, optional): Path to the test data file. Defaults to None.
        output_file (str, optional): Path to the output file for saving the evaluation results. Defaults to "detailed_evaluation.json".
        extraction_agent_model_name (str, optional): Name of the agent model used for extraction. Defaults to "GPT-4o-mini".
        is_synthesis_evaluation (bool, optional): A flag to indicate if synthesis evaluation is required. Defaults to True.
        weights (dict, optional): Weights for the evaluation metrics. Defaults to None.
        llm (LLM, optional): An instance of the LLM class. Defaults to None.
    """
    scanner = ComProScanner(main_property_keyword="placeholder")
    return scanner.evaluate_agentic(
        ground_truth_file=ground_truth_file,
        test_data_file=test_data_file,
        output_file=output_file,
        agent_model_name=extraction_agent_model_name,
        is_synthesis_evaluation=is_synthesis_evaluation,
        weights=weights,
        llm=llm,
    )
