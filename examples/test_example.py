import os
from dotenv import load_dotenv
from comproscanner import ComProScanner
from crewai import LLM

load_dotenv()

flow_optional_args = {
    "composition_property_extraction_task_notes": [
        "Write complete chemical formulas (e.g. '(Use the abbreviation key-value pair to track the abbreviations while extracting composition-property keywords). Don't use abbreviations, but you can use different environment if multiple compositions have different d33 values in different environments. For e.g.: 'LiNbO3-Air' and 'LiNbO3-Vacuum'.",
        "If multiple compositions are mentioned with different d33 values, extract all compositions mentioning different type/environment with a '-' and coreresponding d33 values.",
        "Mos of the formula are provided in ABSTRACT or INTRODUCTION or first part of EXPERIMENTAL SYNTHESIS.",
        "Use element symbols (e.g. 'K' not 'Potassium')",
        "For compositions with variables:\n   * If multiple d33 values exist: Extract all compositions by substituting each variable value\n   * If single d33 value: Extract only the best composition",
        "Prioritize data from tables",
        "Preserve proper chemical notation including:\n   * Subscripts for atom counts\n   * Parentheses for grouped elements\n   * Proper fraction formatting",
        "Include the measurement unit for d33 values (usually pC/N or pm/V)",
        "Identify the broader family/class of materials - can be abbreviated forms",
        "Include if there is any doping in the composition using '+' sign with the composition, e.g. 'LiNbO3 + 2%wt Mg'",
        "Don't remove brackets or parentheses from the chemical formula which is already in the composition. Also, sometimes fractions are written before the composition (mostly in ()) for composition mixtures, don't remove them.",
        "Don't modify the abbreviation from referenced papers unless full composition is not available.",
        "ALWAYS follow the asked JSON format. DON'T write any extra information such as note points or explanations.",
    ],
    "synthesis_extraction_task_notes": [
        "For synthesis_methods, use the short name of the method if possible, else write the full name and don't use method/reaction word at the end.",
        "For precursors, just use the chemical composition (if available) else chemical name (no company or purity)",
        "For characterization_techniques, try using short name of the characterization techniques like XRD, Raman Spectroscopy, SEM, TEM etc. if possible, else write the full name.",
    ],
}

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
        model="deepseek/deepseek-chat",
        output_log_folder="piezo_test/model-logs/logs/deepseek/deepseek-v3-0324",
        task_output_folder="piezo_test/model-logs/task_outputs/deepseek/deepseek-v3-0324",
        materials_data_identifier_query="Is there any ceramic, composite, or crystal material with its specific chemical composition and corresponding d33 piezoelectric coefficient value (in pC/N or pm/V units) explicitly mentioned in the paper? Give one word answer - either 'yes' or 'no'. Only answer 'yes' if ALL of the following criteria are met: (1) The material is specifically a ceramic, composite, doped, or crystal, or different environments of materials (exclude all polymers including PVDF, PLLA, and similar), (2) A numerical d33 value with units pC/N or pm/V is explicitly stated which is associated with that specific material composition or specific environment.",
        json_results_file="piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json",
        **flow_optional_args,
    )

    comproscanner.evaluate_semantic(
        ground_truth_file="piezo_test/ground_truth.json",
        test_data_file="piezo_test/model-outputs/deepseek/deepseek-v3-piezo-ceramic-test-results.json",
        output_file="piezo_test/eval-results/semantic-evaluation/deepseek-v3-0324-semantic-evaluation-results.json",
        agent_model_name="DeepSeek-V3-0324",
    )

    llm = LLM(model="gemini/gemini-2.5-pro")

    comproscanner.evaluate_agentic(
        ground_truth_file="piezo_test/ground_truth.json",
        test_data_file="piezo_test/model-outputs/deepseek/deepseek-v3-piezo-ceramic-test-results.json",
        output_file="piezo_test/eval-results/agentic-evaluation/deepseek-v3-0324-agentic-evaluation-results.json",
        agent_model_name="DeepSeek-V3-0324",
        llm=llm,
    )
