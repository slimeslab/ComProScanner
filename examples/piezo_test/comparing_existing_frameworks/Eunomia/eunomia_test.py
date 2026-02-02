import eunomia
from eunomia.agents import Eunomia
from eunomia.tools import EunomiaTools
import logging
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from eunomia.parser import parse_to_dict
from eunomia.prompts import PIEZO_EXTRACTION_PROMPT

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Redirect stdout and stderr to capture all terminal output in the same log file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


# Open log file for both logging and terminal output
log_file = open("eunomia.log", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Print to console (which is now Tee'd to file)
)

logger = logging.getLogger(__name__)

# Silence all other loggers except our main logger and agent observations
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.CRITICAL)
logging.getLogger("langchain_openai").setLevel(logging.CRITICAL)
logging.getLogger("langchain_community").setLevel(logging.CRITICAL)
logging.getLogger("faiss").setLevel(logging.CRITICAL)

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


def extract_doi_from_filename(filename):
    """Extract DOI from filename by replacing _ with /"""
    doi = filename.replace(".pdf", "").replace("_", "/")
    return doi


def process_single_paper(paper_path, paper_id):
    """Process a single paper and extract piezoelectric data"""

    try:
        # Load and process document silently
        docs_processor = eunomia.LoadDoc(file_name=paper_path, encoding="utf8")
        sliced_pages = docs_processor.process(
            ["references ", "acknowledgement", "acknowledgments", "references\n"],
            chunk_size=1000,
            chunk_overlap=25,
            chunking_type="fixed-size",
        )

        # Create FAISS index silently
        embedding_model = "text-embedding-ada-002"
        faiss_index = FAISS.from_documents(
            sliced_pages,
            OpenAIEmbeddings(model=embedding_model, api_key=OPENAI_API_KEY),
        )

        # Set up tools and agent
        tools = EunomiaTools(
            tool_names=["read_doc", "eval_justification", "recheck_justification"],
            vectorstore=faiss_index,
        ).get_tools()

        agent = Eunomia(
            tools=tools,
            model="deepseek-chat",
            get_cost=False,  # Disable cost tracking to avoid warnings
            agent_type=eunomia.AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        )

        # Run extraction - agent will log its own observations/thoughts
        result = agent.run(prompt=PIEZO_EXTRACTION_PROMPT)

        # Parse result to structured format silently
        try:
            parsed_data = parse_to_dict(result, paper_id)
            num_compositions = len(
                parsed_data.get("composition_data", {}).get(
                    "compositions_property_values", {}
                )
            )
            logger.info(f"✓ Extracted {num_compositions} composition(s)\n")
            return parsed_data
        except Exception as e:
            logger.error(f"✗ Parsing failed: {e}\n")
            return None

    except Exception as e:
        logger.error(f"✗ Processing failed: {e}\n")
        return None


def create_database(
    doi_list, data_folder="data", output_file="piezo_extracted_results.json"
):
    """
    Process specified papers by DOI from the data folder and create a database

    Parameters:
    - doi_list: List of DOIs to process
    - data_folder: Path to folder containing PDF files
    - output_file: Path to output JSON database file

    Returns:
    - dict: Complete database of all extracted compositions
    """
    logger.info(f"{'='*80}")
    logger.info(f"DATABASE CREATION STARTED")
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"{'='*80}\n")

    # Get all PDF files in the data folder
    data_path = Path(data_folder)

    # Convert DOIs to filenames and check if they exist
    pdf_files = []
    for doi in doi_list:
        filename = doi.replace("/", "_") + ".pdf"
        pdf_path = data_path / filename
        if pdf_path.exists():
            pdf_files.append(pdf_path)
        else:
            logger.warning(f"File not found for DOI: {doi}")

    if not pdf_files:
        logger.warning(f"No PDF files found for provided DOIs in {data_folder}")
        return {}

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process\n")

    # Initialise database
    complete_database = {}
    processing_stats = {
        "total_papers": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "total_compositions": 0,
        "failed_papers": [],
    }

    # Process each paper
    for idx, pdf_file in enumerate(pdf_files, 1):
        paper_filename = pdf_file.name
        doi = extract_doi_from_filename(paper_filename)

        logger.info(f"{'='*80}")
        logger.info(f"Paper {idx}/{len(pdf_files)}: {doi}")
        logger.info(f"{'='*80}")

        try:
            # Process paper
            paper_data = process_single_paper(str(pdf_file), doi)

            if paper_data and paper_data.get("composition_data", {}).get(
                "compositions_property_values"
            ):
                # Add to database with paper DOI as key
                complete_database[doi] = paper_data
                processing_stats["successful"] += 1
                num_compositions = len(
                    paper_data["composition_data"]["compositions_property_values"]
                )
                processing_stats["total_compositions"] += num_compositions
            else:
                processing_stats["failed"] += 1
                processing_stats["failed_papers"].append(doi)

        except Exception as e:
            processing_stats["failed"] += 1
            processing_stats["failed_papers"].append(doi)
            logger.error(f"✗ Error: {e}\n")
            continue

    # Save database to JSON file
    output_path = Path(output_file)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(complete_database, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving database: {e}")

    # Print summary statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total papers: {processing_stats['total_papers']}")
    logger.info(f"Successful: {processing_stats['successful']}")
    logger.info(f"Failed: {processing_stats['failed']}")
    logger.info(f"Total compositions: {processing_stats['total_compositions']}")
    logger.info(f"{'='*80}\n")

    if processing_stats["failed_papers"]:
        logger.warning("Failed papers:")
        for failed_doi in processing_stats["failed_papers"]:
            logger.warning(f"  - {failed_doi}")

    return complete_database


if __name__ == "__main__":
    with open("../selected_dois.txt", "r", encoding="utf-8") as f:
        doi_list = [line.strip() for line in f.readlines() if line.strip()]
    database = create_database(
        doi_list=doi_list,
        data_folder="data",
        output_file="piezo_extracted_results.json",
    )

    # Close log file at the end
    log_file.close()
