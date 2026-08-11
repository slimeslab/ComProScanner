"""
run_piezo_agent.py

Main LangGraph workflow for piezoelectric materials extraction.
Modified to use DeepSeek API and output single JSON file with DOI keys.
Includes verbose logging of all agent actions and LLM calls.
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import TypedDict, Literal
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from piezo_agent_tools import (
    extract_material_candidates,
    extract_piezo_properties,
    extract_synthesis_properties,
    extract_from_tables,
    merge_extraction_results,
)

load_dotenv()

# === Configure Logging ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("CMEG-IITR-Agent_test.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# === DeepSeek Configuration ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
model_name = "deepseek-v4-flash"


# === State Definition ===
class State(TypedDict):
    folder: Path
    doi: str
    fulltext: str
    llm: any
    material_names: list
    piezo: dict
    synthesis: dict
    retries: int
    skip: bool
    table_data: list
    table_json_output: dict
    total_table_rows: int
    merged_result: dict


# === Node Functions ===
def read_file_node(state: State) -> State:
    logger.info(f"{'='*80}")
    logger.info(f"NODE: read_file")
    logger.info(f"{'='*80}")

    folder = state["folder"]
    fulltext_path = folder / "fulltext.txt"

    if not fulltext_path.exists():
        logger.error(f"❌ No fulltext.txt in {folder.name}")
        return {**state, "skip": True}

    with open(fulltext_path, "r", encoding="utf-8") as f:
        fulltext = f.read()

    # Extract DOI from folder name (convert _ back to /)
    doi = folder.name.replace("_", "/")

    logger.info(f"📄 Read {len(fulltext)} chars from {folder.name}")
    logger.info(f"📎 DOI: {doi}")
    logger.debug(f"Fulltext preview: {fulltext[:500]}...")

    return {**state, "fulltext": fulltext, "doi": doi, "skip": False}


def set_tokens_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: set_tokens")
    logger.info(f"{'='*80}")

    folder = state["folder"]
    token_file = folder / "token_count.txt"

    token_count = 999
    if token_file.exists():
        with open(token_file, "r") as f:
            token_count = int(f.read().strip())

    if token_count == 0:
        logger.warning(f"⏭️ Skipping {folder.name} due to token_count = 0")
        return {**state, "skip": True}

    # Compute max_tokens
    if token_count <= 1000:
        max_tok = 786
    else:
        extra = (token_count - 1000) // 500
        max_tok = 786 + (256 * extra)
        max_tok = min(max_tok, 8192)

    logger.info(f"🧠 Token count: {token_count}")
    logger.info(f"🧠 Setting max_tokens = {max_tok} for {folder.name}")

    dynamic_llm = ChatOpenAI(
        model=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.001,
        max_tokens=max_tok,
        verbose=True,
    )

    logger.debug(
        f"LLM configured: model={model_name}, max_tokens={max_tok}, temperature=0.001"
    )

    return {**state, "llm": dynamic_llm, "skip": False}


def find_materials_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: find_materials")
    logger.info(f"{'='*80}")

    small_llm = ChatOpenAI(
        model=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.001,
        max_tokens=256,
        verbose=True,
    )

    logger.info("🔍 Calling extract_material_candidates...")
    logger.debug(f"Input text length: {len(state['fulltext'])} chars")

    candidates = extract_material_candidates(
        state["fulltext"], llm=small_llm, max_materials=20
    )

    if candidates:
        logger.info(f"🧪 Candidate materials found: {len(candidates)}")
        logger.info(
            f"📋 Candidates: {candidates[:5]}{'...' if len(candidates) > 5 else ''}"
        )
        logger.debug(f"Full candidate list: {candidates}")
        return {**state, "material_names": candidates, "skip": False}
    else:
        logger.warning("🛑 No piezoelectric materials found → skipping")
        return {**state, "material_names": [], "skip": True}


def extract_piezo_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: extract_piezo")
    logger.info(f"{'='*80}")

    logger.info("🔬 Extracting piezoelectric properties...")
    logger.debug(f"Material hints: {state.get('material_names', [])}")

    piezo = extract_piezo_properties(
        state["fulltext"], llm=state["llm"], material_names=state.get("material_names")
    )

    logger.info(f"✅ Piezo extraction complete")
    logger.debug(f"Extracted piezo data: {json.dumps(piezo, indent=2)}")

    # Log extracted compositions
    if piezo.get("composition_data", {}).get("compositions_property_values"):
        comps = piezo["composition_data"]["compositions_property_values"]
        logger.info(f"📊 Extracted {len(comps)} compositions with d33 values")
        for comp, value in list(comps.items())[:5]:
            logger.info(f"  - {comp}: {value}")

    return {**state, "piezo": piezo}


def extract_synthesis_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: extract_synthesis")
    logger.info(f"{'='*80}")

    logger.info("🧪 Extracting synthesis properties...")
    logger.debug(f"Material hints: {state.get('material_names', [])}")

    synthesis = extract_synthesis_properties(
        state["fulltext"], llm=state["llm"], material_names=state.get("material_names")
    )

    logger.info(f"✅ Synthesis extraction complete")
    logger.debug(f"Extracted synthesis data: {json.dumps(synthesis, indent=2)}")

    # Log synthesis details
    if synthesis.get("synthesis_data"):
        synth_data = synthesis["synthesis_data"]
        logger.info(f"🔧 Method: {synth_data.get('method', 'N/A')}")
        logger.info(f"🧪 Precursors: {len(synth_data.get('precursors', []))} found")
        logger.info(f"📝 Steps: {len(synth_data.get('steps', []))} found")
        logger.info(
            f"📊 Characterization: {len(synth_data.get('characterization_techniques', []))} techniques"
        )

    return {**state, "synthesis": synthesis}


def count_table_and_plan_tokens_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: count_table_and_plan_tokens")
    logger.info(f"{'='*80}")

    folder = state["folder"]
    table_data = []
    total_rows = 0
    i = 1

    while True:
        csv_path = folder / f"table{i}.csv"
        caption_path = folder / f"table{i}_caption.txt"
        if not csv_path.exists() or not caption_path.exists():
            break
        try:
            df = pd.read_csv(csv_path)
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            row_count = len(df)

            logger.debug(f"📋 Table {i}: {row_count} rows, caption: {caption[:100]}...")

            table_data.append(
                {
                    "filename": f"table{i}.csv",
                    "caption": caption,
                    "rows": df.to_dict(orient="records"),
                    "row_count": row_count,
                }
            )
            total_rows += row_count
        except Exception as e:
            logger.warning(f"⚠️ Failed reading {csv_path.name}: {e}")
        i += 1

    max_tokens = min(512 + total_rows * 128, 8192) if total_rows > 0 else 512

    dynamic_llm = ChatOpenAI(
        model=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=max_tokens,
        verbose=True,
    )

    logger.info(f"📊 Found {len(table_data)} tables with {total_rows} total rows")
    logger.info(f"🧠 Adjusted max_tokens = {max_tokens} for table extraction")

    return {
        **state,
        "table_data": table_data,
        "total_table_rows": total_rows,
        "llm": dynamic_llm,
    }


def extract_table_json_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: extract_table_json")
    logger.info(f"{'='*80}")

    if not state.get("table_data"):
        logger.info("⏭️ No tables to process")
        return {
            **state,
            "table_json_output": {
                "composition_data": {
                    "compositions_property_values": {},
                    "property_unit": "",
                    "family": "",
                }
            },
        }

    logger.info(f"📊 Extracting data from {len(state['table_data'])} tables...")

    table_json = extract_from_tables(
        state["table_data"],
        llm=state["llm"],
        material_names=state.get("material_names"),
    )

    logger.info(f"✅ Table extraction complete")
    logger.debug(f"Extracted table data: {json.dumps(table_json, indent=2)}")

    # Log table extraction results
    if table_json.get("composition_data", {}).get("compositions_property_values"):
        comps = table_json["composition_data"]["compositions_property_values"]
        logger.info(f"📋 Extracted {len(comps)} compositions from tables")

    return {**state, "table_json_output": table_json}


def merge_results_node(state: State) -> State:
    logger.info(f"\n{'='*80}")
    logger.info(f"NODE: merge_results")
    logger.info(f"{'='*80}")

    logger.info("🔄 Merging extraction results...")

    merged_result = merge_extraction_results(
        piezo_data=state.get("piezo", {}),
        synthesis_data=state.get("synthesis", {}),
        table_data=state.get("table_json_output", {}),
    )

    logger.info(f"✅ Merge complete for {state['doi']}")
    logger.debug(f"Merged result: {json.dumps(merged_result, indent=2)}")

    # Log final statistics
    num_comps = len(
        merged_result.get("composition_data", {}).get(
            "compositions_property_values", {}
        )
    )
    num_precursors = len(merged_result.get("synthesis_data", {}).get("precursors", []))
    num_steps = len(merged_result.get("synthesis_data", {}).get("steps", []))

    logger.info(f"📊 Final counts:")
    logger.info(f"  - Compositions: {num_comps}")
    logger.info(f"  - Precursors: {num_precursors}")
    logger.info(f"  - Synthesis steps: {num_steps}")

    return {**state, "merged_result": merged_result}


# === Routing Functions ===
def skip_or_continue(state: State) -> Literal["end", "find_materials"]:
    route = "end" if state.get("skip") else "find_materials"
    logger.debug(f"Routing decision after read_file: {route}")
    return route


def skip_or_extract(state: State) -> Literal["end", "extract_piezo"]:
    route = "end" if state.get("skip") else "extract_piezo"
    logger.debug(f"Routing decision after find_materials: {route}")
    return route


# === Build Graph ===
logger.info("Building LangGraph workflow...")

workflow = StateGraph(State)

workflow.add_node("read_file", read_file_node)
workflow.add_node("set_tokens", set_tokens_node)
workflow.add_node("find_materials", find_materials_node)
workflow.add_node("extract_piezo", extract_piezo_node)
workflow.add_node("extract_synthesis", extract_synthesis_node)
workflow.add_node("count_table_tokens", count_table_and_plan_tokens_node)
workflow.add_node("extract_tables", extract_table_json_node)
workflow.add_node("merge_results", merge_results_node)

workflow.set_entry_point("read_file")
workflow.add_conditional_edges(
    "read_file", skip_or_continue, {"end": END, "find_materials": "set_tokens"}
)
workflow.add_edge("set_tokens", "find_materials")
workflow.add_conditional_edges(
    "find_materials", skip_or_extract, {"end": END, "extract_piezo": "extract_piezo"}
)
workflow.add_edge("extract_piezo", "extract_synthesis")
workflow.add_edge("extract_synthesis", "count_table_tokens")
workflow.add_edge("count_table_tokens", "extract_tables")
workflow.add_edge("extract_tables", "merge_results")
workflow.add_edge("merge_results", END)

app = workflow.compile()

logger.info("✅ LangGraph workflow compiled successfully")


# === Main Execution ===
if __name__ == "__main__":
    base_dir = Path("elsevier_piezo_processed")
    output_file = "piezo_extracted_results.json"

    logger.info(f"\n{'#'*80}")
    logger.info(f"STARTING PIEZOELECTRIC EXTRACTION PIPELINE")
    logger.info(f"{'#'*80}\n")

    if not base_dir.exists():
        logger.error(
            f"❌ Directory {base_dir} does not exist. Run piezo_data_preprocessing.py first."
        )
        exit(1)

    folders = sorted([f for f in base_dir.iterdir() if f.is_dir()])
    logger.info(f"📂 Found {len(folders)} folders to process")
    logger.info(f"🎯 Processing first 10 folders for testing\n")

    # Initialize database dictionary
    complete_database = {}
    processing_stats = {
        "total_papers": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total_compositions": 0,
        "failed_dois": [],
    }

    for idx, folder in enumerate(folders[:10], 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"PROCESSING PAPER {idx}/{len(folders[:10])}: {folder.name}")
        logger.info(f"{'#'*80}\n")

        processing_stats["total_papers"] += 1

        try:
            result_state = app.invoke(
                State(
                    folder=folder,
                    doi="",
                    fulltext=None,
                    llm=None,
                    material_names=None,
                    piezo=None,
                    synthesis=None,
                    retries=0,
                    skip=False,
                    table_data=None,
                    table_json_output=None,
                    total_table_rows=0,
                    merged_result=None,
                )
            )

            # Check if processing was skipped
            if result_state.get("skip"):
                logger.warning(f"⏭️ Skipped {folder.name}")
                processing_stats["skipped"] += 1
                continue

            # Check if we have valid results
            if result_state.get("merged_result") and result_state.get("doi"):
                doi = result_state["doi"]
                merged_data = result_state["merged_result"]

                # Add to database
                complete_database[doi] = merged_data

                # Update statistics
                num_compositions = len(
                    merged_data.get("composition_data", {}).get(
                        "compositions_property_values", {}
                    )
                )
                processing_stats["successful"] += 1
                processing_stats["total_compositions"] += num_compositions

                logger.info(f"✅ Successfully completed {folder.name}")
                logger.info(f"📊 Extracted {num_compositions} compositions")
            else:
                logger.warning(f"⚠️ No valid results for {folder.name}")
                processing_stats["failed"] += 1
                processing_stats["failed_dois"].append(folder.name)

            # Rate limiting
            t = random.uniform(6, 10)
            logger.debug(f"⏱️ Sleeping for {t:.1f} seconds (rate limiting)...")
            time.sleep(t)

        except Exception as e:
            logger.error(f"❌ Failed {folder.name}: {e}", exc_info=True)
            processing_stats["failed"] += 1
            processing_stats["failed_dois"].append(folder.name)

    # Save complete database to single JSON file
    logger.info(f"\n{'#'*80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'#'*80}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(complete_database, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved complete database to: {output_file}")
    logger.info(f"💾 Database size: {len(complete_database)} papers")

    # Print summary statistics
    logger.info(f"\n{'#'*80}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'#'*80}")
    logger.info(f"Total papers processed: {processing_stats['total_papers']}")
    logger.info(f"✅ Successful: {processing_stats['successful']}")
    logger.info(f"⏭️ Skipped: {processing_stats['skipped']}")
    logger.info(f"❌ Failed: {processing_stats['failed']}")
    logger.info(f"📊 Total compositions: {processing_stats['total_compositions']}")
    if processing_stats["successful"] > 0:
        avg = processing_stats["total_compositions"] / processing_stats["successful"]
        logger.info(f"📈 Average compositions/paper: {avg:.1f}")
    logger.info(f"{'#'*80}\n")

    if processing_stats["failed_dois"]:
        logger.warning("Failed DOIs:")
        for failed_doi in processing_stats["failed_dois"]:
            logger.warning(f"  - {failed_doi}")

    logger.info("\n🎉 All processing complete!")
    logger.info(f"📋 Complete log saved to: piezo_agent.log")
