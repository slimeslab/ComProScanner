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
        "IMPORTANT: If a single-phase perovskite solid solution or single crystal is confirmed after doping (look for keywords: 'single crystal', 'solid solution', 'single phase', 'perovskite structure retained', 'XRD confirms single phase'), derive a charge-balanced merged single-compound ABO3 formula using the following steps:\n   1. Identify the dopant site (A or B) by comparing ionic radius and valence to host cations.\n   Determine the charge compensation mechanism from the text (A-site vacancies, B-site vacancies, or oxygen non-stoichiometry).\n   Scale all existing occupancies on the substituted site by (1-x) and insert the dopant term with its charge-balance-derived prefactor.\n   Write oxygen always as O3 (or O(3-delta) if oxygen vacancy compensation is stated).\n   Merge everything into a single ABO3 expression with NO '+' sign and NO ':' notation.\n   NEVER use the colon notation (e.g. 'PbZr0.52Ti0.48O3:La3+') for confirmed single-phase or single-crystal systems. That notation is reserved ONLY for dopants where site substitution cannot be determined.\n   Use plain-text variable notation with disambiguating parentheses (no underscore subscripts), and ALWAYS include explicit '*' for multiplication in variable expressions. For example, write 0.5*(1-x-y), 0.5*x, and 2*x (never 0.5(1-x-y), 0.5x, or 2x). Example (A-site donor doping with A-site vacancy compensation): x mol% La3+ doped PbZr0.52Ti0.48O3 gives (Pb(1-x)La(x/2))(Zr0.52Ti0.48)O3 if La occupies the A-site with Pb vacancies as compensation.",
        "If variable values are given in mol% or at% or wt%, convert them to decimal fractions before substituting  into the formula (divide by 100). For example, x=0.5 mol% becomes x=0.005, so Ti(1-x) becomes Ti0.995 and Fex becomes Fe0.005. NEVER substitute the raw percentage number directly."
        "Include the measurement unit for d33 values (usually pC/N or pm/V)",
        "Identify the broader family/class of materials - can be abbreviated forms",
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

PEROVSKITE_FORMULA_INSTRUCTION = """
You are a materials science expert specializing in perovskite ceramics and solid-state chemistry.

You will be given a research paper. Your job is to:
1. Determine if doping produces a single-phase perovskite compound.
2. If yes, derive the general doped chemical formula.
3. If no, return exactly: single compound is not being synthesized

---

STEP 1 - CHECK FOR SINGLE PHASE FORMATION

Look for:
- XRD evidence of a single perovskite phase with no secondary phases
- Language like "pure perovskite phase", "single phase", "complete solid solution"
- No reported phase separation or multiphase coexistence

If secondary phases, phase separation or multiphase coexistence is reported, return:
single compound is not being synthesized

---

STEP 2 - IDENTIFY BASE COMPOSITION AND DOPANT

Extract:
- A-site ions and their mole fractions in the undoped base
- B-site ions and their mole fractions in the undoped base
- Dopant identity and concentration variable (commonly x or mol%)
- Which site the dopant substitutes (A-site or B-site)
- How the dopant concentration is reported: mol%, at%, wt% or direct mole fraction

---

STEP 3 - CONVERT DOPANT CONCENTRATION TO MOLE FRACTION

Apply the correct conversion based on how the dopant is reported in the paper:

IF reported as mol% or at%:
- These are equivalent to mole fraction in ceramic compositions.
- Use x directly as the mole fraction of the dopant on its sublattice.

IF reported as wt%:
- Convert wt% to mol fraction before proceeding.
- mol fraction of dopant = (wt% / M_dopant) / sum over all components of (wt%_i / M_i)
- where M is the molar mass of each component oxide or compound as used in the synthesis.
- Use the converted mol fraction as x in the formula.

IF reported as direct mole fraction:
- Use x directly with no conversion.

---

STEP 4 - CONSTRUCT THE FORMULA

Rules:
1. Scale all original A-site coefficients by (1-x) to reflect dilution by the dopant.
2. Scale all original B-site coefficients by (1-x) if the dopant is on the B-site, otherwise keep them unchanged.
3. Insert the dopant term on the correct site with coefficient x (or the converted mol fraction from Step 3).
4. Always write oxygen stoichiometry as O3.
5. Do NOT use + or - signs anywhere in the formula.
6. Do NOT write vacancy symbols such as Va or [] in the formula.
7. Write in standard ABO3 perovskite notation: (A-site terms)(B-site terms)O3
8. All coefficients must be explicit algebraic expressions in x.
9. Use plain-text coefficients without underscore subscripts.
10. Add parentheses around variable expressions where scope may be ambiguous.
11. Always use explicit '*' for multiplication in coefficient expressions. Examples: 0.5*(1-x-y), 0.5*x, 2*x.

---

OUTPUT FORMAT

Return one of the following and nothing else:

- If single phase forms: return only the plain text chemical formula with no LaTeX, no + or - signs,
  no vacancy notation and no explanation.
  Example: (Na0.53*(1-x)K0.404*(1-x)Li0.066*(1-x)Ca(x))(Nb0.92*(1-x)Sb0.08*(1-x)Al(2*x))O3

- If not single phase: return exactly the string:
  single compound is not being synthesized
"""

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
        formula_instruction=PEROVSKITE_FORMULA_INSTRUCTION,
        equation_model="anthropic/claude-sonnet-4-6",
        checked_doi_list_file="checked_dois_gemini-3-flash-preview.txt",
        **flow_optional_args,
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
