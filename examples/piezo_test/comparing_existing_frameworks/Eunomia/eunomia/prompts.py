RULES = """
There are specific requirements for piezoelectric composition extraction:
1. Composition: Extract the complete chemical formula or material name as written in the paper.
   Include all dopant concentrations, stoichiometric ratios, and chemical notation.
   Example: (1-x)Ba(Zr0.2Ti0.8)O3-x(Ba0.7Ca0.3)TiO3 or Pb(Zr0.52Ti0.48)O3

2. d33 Value: Extract the piezoelectric charge coefficient (d33) value with its numerical magnitude.
   Report the maximum or representative value if multiple measurements are provided.
   
3. Unit: Extract the unit of measurement for d33 (typically pC/N or pm/V).

4. Composition Family: Identify the broader material family or system.
   Examples: BaTiO3-based, PZT-based, KNN-based, BCZT, relaxor ferroelectrics

5. Synthesis Method: Extract the primary preparation or synthesis route.
   Examples: solid-state reaction, sol-gel, hydrothermal, spark plasma sintering

6. If information is not explicitly provided in the document, report as "Not provided".
"""

PIEZO_EXTRACTION_PROMPT = f"""
    You are an expert materials scientist specialising in piezoelectric materials. The document describes 
    piezoelectric materials, their compositions, properties, and synthesis procedures.
    
    Extract the following information for each piezoelectric composition mentioned:
    1. Complete chemical composition or formula
    2. d33 value (piezoelectric charge coefficient)
    3. Unit of d33 measurement
    4. Composition family or material system
    5. Synthesis method
    6. Precursors used (as a list)
    7. Synthesis steps (as a list of sequential procedures)
    8. Characterisation techniques employed (as a list)
    
    As an example, the sentence "The 0.5Ba(Zr0.2Ti0.8)O3-0.5(Ba0.7Ca0.3)TiO3 ceramic prepared by 
    solid-state reaction exhibits a d33 of 620 pC/N" contains:
    - Composition: 0.5Ba(Zr0.2Ti0.8)O3-0.5(Ba0.7Ca0.3)TiO3
    - d33 value: 620
    - Unit: pC/N
    - Composition family: BCZT
    - Synthesis method: solid-state reaction
    
    Use this example to guide yourself in finding similar information in the document.
    If the paper does not provide specific information for any field, report it as "Not provided".
    Report the paper's DOI.
    
    Use the following rules to determine extraction completeness:
    {RULES}
    
    Your final answer should contain the following for each composition:
    1. The complete chemical composition exactly as written in the document.
    2. The d33 value and its corresponding unit.
    3. The composition family or material system.
    4. The synthesis method employed.
    5. A list of precursors used in synthesis. This should be "Not provided" if you cannot find.
    6. A list of synthesis steps in sequential order. This should be "Not provided" if you cannot find.
    7. A list of characterisation techniques used. This should be "Not provided" if you cannot find.
    8. For each extracted composition: a probability score ranging between [0, 1]. This probability score 
       shows how certain you are in your extraction accuracy.
    9. The exact sentences without any changes from the document that justify your extraction. Try to find 
       more than one sentence where relevant.
    10. Paper's DOI. This should be "Not provided" if you cannot find.
    """