import os
from dotenv import load_dotenv
from comproscanner import ComProScanner
from crewai import LLM

load_dotenv()

main_property_keyword = "piezoelectric"
property_keywords = {"exact_keywords": ["d33"], "substring_keywords": [" d 33 "]}


if __name__ == "__main__":
    base_queries = [
        "piezoelectric",
        "piezoelectricity",
        "pyroelectric",
        "pyroelectricity",
        "ferroelectric",
        "ferroelectricity",
    ]
    extra_queries = [
        "advancements",
        "applications",
        "ceramics",
        "characterization",
        "composites",
        "crystals",
        "devices",
        "doped",
        "doping",
        "enhancement",
        "fabrication",
        "integration",
        "materials",
        "nanomaterials",
        "optimization",
        "properties",
        "sensor",
        "techniques",
    ]
    comproscanner = ComProScanner(main_property_keyword=main_property_keyword)
    comproscanner.collect_metadata(
        base_queries=base_queries,
        extra_queries=extra_queries,
        end_year=2019,
    )

    comproscanner.process_articles(
        property_keywords=property_keywords,
        source_list=["elsevier"],
    )

    comproscanner.extract_composition_property_data(
        main_extraction_keyword="d33",
        is_test_data_preparation=True,
        # is_only_consider_test_doi_list=True,
        test_doi_list_file="piezo_test_dois_random.txt",
        total_test_data=100,
        model="deepseek/deepseek-v4-flash",
        output_log_folder="piezo_test/model-logs/logs/deepseek/deepseek-v3-0324",
        task_output_folder="piezo_test/model-logs/task_outputs/deepseek/deepseek-v3-0324",
        materials_data_identifier_query="Is there any ceramic, composite, or crystal material with its specific chemical composition and corresponding d33 piezoelectric coefficient value (in pC/N or pm/V units) explicitly mentioned in the paper? Give one word answer - either 'yes' or 'no'. Only answer 'yes' if ALL of the following criteria are met: (1) The material is specifically a ceramic, composite, doped, or crystal, or different environments of materials (exclude all polymers including PVDF, PLLA, and similar), (2) A numerical d33 value with units pC/N or pm/V is explicitly stated which is associated with that specific material composition or specific environment.",
        json_results_file="piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json",
        instructions_config_path="piezo_instructions.yaml",
    )

    comproscanner.evaluate_semantic(
        ground_truth_file="piezo_test/ground_truth.json",
        test_data_file="piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json",
        output_file="piezo_test/eval-results/semantic-evaluation/deepseek-v3-0324-semantic-evaluation-results.json",
        extraction_agent_model_name="DeepSeek-V3-0324",
    )

    llm = LLM(model="gemini/gemini-2.5-pro")

    comproscanner.evaluate_agentic(
        ground_truth_file="piezo_test/ground_truth.json",
        test_data_file="piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json",
        output_file="piezo_test/eval-results/agentic-evaluation/deepseek-v3-0324-agentic-evaluation-results.json",
        extraction_agent_model_name="DeepSeek-V3-0324",
        llm=llm,
    )
