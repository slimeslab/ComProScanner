import os
from dotenv import load_dotenv
from comproscanner import ComProScanner
from crewai import LLM

load_dotenv()

main_property_keyword = "piezoelectric"
property_keywords = {
    "exact_keywords": ["d33", "pC/N"],
    "substring_keywords": [" d 33 ", " pC/N "],
}
main_figure_keywords = {
    "exact_keywords": [
        "d33",
        "piezoelectric",
        "pC/N",
        "xrd",
        "x-ray diffraction",
        "x-ray",
        "diffraction",
        "sem",
        "scanning electron",
        "tem",
        "transmission electron",
        "crystal structure",
        "microstructure",
    ],
    "substring_keywords": [
        " d 33 ",
        " piezoelectric ",
        " pC/N ",
        " xrd ",
        " x-ray diffraction ",
        " x-ray ",
        " diffraction ",
        " sem ",
        " scanning electron ",
        " tem ",
        " transmission electron ",
        " crystal structure ",
        " microstructure ",
    ],
}

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

    # get all the test DOIs from random_dois_for_vlm_test.txt file.
    with open("random_dois_for_vlm_test.txt", "r") as f:
        test_doi_list = [line.strip() for line in f.readlines()]

    comproscanner = ComProScanner(main_property_keyword=main_property_keyword)

    # NOTE: For test data preparation, we skipped the metadata collection step and used already collected metadata from previeous test.
    comproscanner.collect_metadata(
        base_queries=base_queries,
        extra_queries=extra_queries,
        end_year=2019,
    )

    comproscanner.process_articles(
        property_keywords=property_keywords,
        main_figure_keywords=main_figure_keywords,
        source_list=["elsevier", "springer", "iop", "wiley"],
        doi_list=test_doi_list,
    )

    comproscanner.extract_composition_property_data(
        main_extraction_keyword="d33",
        is_only_consider_test_doi_list=True,
        test_doi_list_file="random_dois_for_vlm_test.txt",
        is_extract_synthesis_data=False,  # For this test, we are only evaluating the composition-property extraction capability of the VLM, so we set this to False to save time and cost.
        model="deepseek/deepseek-v4-flash",
        vlm_model="gemini/gemini-3-flash-preview",
        output_log_folder="vlm_piezo_test/model-logs/logs/google/gemini-3-flash-preview",
        task_output_folder="vlm_piezo_test/model-logs/task_outputs/google/gemini-3-flash-preview",
        materials_data_identifier_query="Is there any ceramic, composite, or crystal material with its chemical composition or doping data, and corresponding d33 piezoelectric coefficient value (in pC/N or pm/V units) mentioned in the text of this paper? Give one word answer - either 'yes' or 'no'. Only answer 'yes' if ALL of the following criteria are met: (1) The material is specifically a ceramic, composite, doped, or crystal, or different environments of materials (exclude all polymers including PVDF, PLLA, and similar), (2) A numerical d33 value with units pC/N or pm/V is explicitly stated, associated with either a specific material composition/environment or a doping concentration variable (e.g. x mol%, at%) — note that in figures/graphs, d33 values plotted against doping concentrations or composition variables also count as relevant data.",
        json_results_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-results.json",
        tools_config_path="piezo_tools.yaml",
        equation_model="anthropic/claude-sonnet-4-6",
        checked_doi_list_file="checked_dois_gemini-3-flash-preview.txt",
        instructions_config_path="piezo_instructions.yaml",
    )

    comproscanner.clean_data(
        json_results_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-results.json",
        cleaned_json_results_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-results-cleaned.json",
        # is_save_composition_property_file=False,
        composition_property_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-composition-property-data.json",
        is_store_unresolved_compositions=True,
        unresolved_compositions_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-unresolved-compositions.json",
    )

    comproscanner.evaluate_semantic(
        ground_truth_file="vlm_piezo_test/vlm-piezo-ceramic-ground-truth.json",
        test_data_file="vlm_piezo_test/model-outputs/google/gemini-3-flash-preview/gemini-3-flash-preview-vlm-piezo-ceramic-test-results-cleaned.json",
        output_file="vlm_piezo_test/eval-results/gemini-3-flash-preview-vlm-semantic-evaluation-results.json",
        extraction_agent_model_name="Gemini-3-Flash-Preview",
        is_synthesis_evaluation=False,
        weights={
            "compositions_property_values": 1.0,
            "property_unit": 0.0,
            "family": 0.0,
            "method": 0.0,
            "precursors": 0.0,
            "characterization_techniques": 0.0,
            "steps": 0.0,
        },
        similarity_thresholds={"compositions_property_values": 1.0},
        value_error_thresholds={
            (-50, 50): 0.5,
            (-150, 150): 1,
            (float("-inf"), float("inf")): 2,
        },
    )
