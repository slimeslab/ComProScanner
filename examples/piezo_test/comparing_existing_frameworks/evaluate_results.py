import os
from dotenv import load_dotenv
from comproscanner import ComProScanner
from crewai import LLM

load_dotenv()

main_property_keyword = "piezoelectric"


if __name__ == "__main__":
    comproscanner = ComProScanner(main_property_keyword=main_property_keyword)

    llm = LLM(model="gemini/gemini-3-flash-preview")

    # comproscanner.evaluate_agentic(
    #     ground_truth_file="comparison_ground_truth.json",
    #     test_data_file="Eunomia/piezo_extracted_results.json",
    #     output_file="Eunomia/piezo_evaluation_results.json",
    #     extraction_agent_model_name="DeepSeek-V3.2",
    #     llm=llm,
    # )

    comproscanner.evaluate_agentic(
        ground_truth_file="comparison_ground_truth.json",
        test_data_file="CMEG-IITR_Agentic_data_extraction/piezo_extracted_results.json",
        output_file="CMEG-IITR_Agentic_data_extraction/piezo_evaluation_results.json",
        extraction_agent_model_name="DeepSeek-V3.2",
        llm=llm,
    )

    # comproscanner.evaluate_agentic(
    #     ground_truth_file="comparison_ground_truth.json",
    #     test_data_file="ComProScanner/comparison_deepseek_results.json",
    #     output_file="ComProScanner/comparison_deepseek_evaluation_results.json",
    #     extraction_agent_model_name="DeepSeek-V3-0324",
    #     llm=llm,
    # )
