"""
create_kg.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 08-04-2025
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging
from typing import Dict
from dotenv import load_dotenv
import os

from ...utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()

######## logger Configuration ########
logger = setup_logger("post-processing.log")


class CreateKG:
    def __init__(self):
        """Initialize Neo4j connection using environment variables"""
        self.driver = None
        try:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            database = os.getenv("NEO4J_DATABASE")

            if not all([uri, user, password]):
                raise ValueError(
                    "Missing required environment variables. Please check your .env file."
                )

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = database
        except ServiceUnavailable as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise
        except ValueError as e:
            logging.error(str(e))
            raise

    def close(self):
        """Close the driver connection"""
        if self.driver is not None:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_paper_with_compositions(
        self,
        synthesis_data: Dict,
        composition_data: Dict,
        paper_metadata: Dict,
    ) -> bool:
        """
        Create paper node with authors, affiliations, and compositions, and establish relationships

        Args:
            synthesis_data: Dictionary containing synthesis information
            composition_data: Dictionary containing composition information
            paper_metadata: Dictionary containing paper information
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Use empty strings instead of None for properties used in MERGE
                family_name = composition_data.get("family", "UNKNOWN_FAMILY")
                method_name = synthesis_data.get("method", "UNKNOWN_METHOD")

                synthesis_steps = (
                    "\n".join(f"- {step}" for step in synthesis_data.get("steps", []))
                    or "No steps available"
                )

                params = {
                    "synthesis_data": synthesis_data or {},
                    "composition_data": composition_data or {},
                    "paper_metadata": paper_metadata,
                    "synthesis_steps": synthesis_steps,
                    "family_name": family_name,
                    "method_name": method_name,
                    "compositions": composition_data.get(
                        "compositions_property_values", {}
                    ),
                    "property_unit": composition_data.get("property_unit"),
                    "precursors": synthesis_data.get("precursors", []),
                    "characterization_techniques": synthesis_data.get(
                        "characterization_techniques", []
                    ),
                }

                query = """
                // Create or find family node with unique name
                MERGE (f:Family {name: $family_name})
                
                // Create or find paper node based on unique DOI
                MERGE (p:Paper {doi: $paper_metadata.doi})
                SET p.title = $paper_metadata.title,
                    p.journal = $paper_metadata.journal,
                    p.year = $paper_metadata.year,
                    p.isOpenAccess = $paper_metadata.isOpenAccess
                
                // Create or find author nodes and their affiliations
                WITH p
                UNWIND $paper_metadata.authors as author
                MERGE (a:Author {name: author.name})
                SET a.affiliation_id = author.affiliation_id
                
                // Create affiliation node and set properties
                MERGE (aff:Affiliation {affiliation_id: author.affiliation_id})
                SET aff.name = author.affiliation_name,
                    aff.country = author.affiliation_country
                
                // Create relationships
                MERGE (a)-[:AFFILIATED_WITH]->(aff)
                MERGE (a)-[:WROTE]->(p)
                
                // Create or find method node based on unique name
                WITH p
                MERGE (m:Method {name: $method_name})
                MERGE (p)-[:USES_METHOD]->(m)
                
                // Create or find step node based on unique steps
                WITH p
                MERGE (st:Step {steps: $synthesis_steps})
                MERGE (p)-[:USED_SYNTHESIS_STEPS]->(st)
                
                // Create relationships between paper and family
                WITH p
                MATCH (f:Family {name: $family_name})
                MERGE (p)-[:BELONGS_TO_FAMILY]->(f)
                
                RETURN p
                """

                result = session.run(query, **params)
                result.consume()

                # Handle compositions separately
                if composition_data.get("compositions_property_values"):
                    compositions_query = """
                    MATCH (p:Paper {doi: $paper_metadata.doi})
                    MATCH (f:Family {name: $family_name})
                    
                    WITH p, f
                    UNWIND keys($compositions) as comp_name
                    MERGE (c:Composition {composition: comp_name})
                    SET c.property_value = $compositions[comp_name],
                        c.property_unit = $property_unit
                    MERGE (c)-[:BELONGS_TO]->(f)
                    MERGE (p)-[:MENTIONS]->(c)
                    """

                    session.run(compositions_query, **params)

                # Handle precursors separately
                if synthesis_data.get("precursors"):
                    precursors_query = """
                    MATCH (p:Paper {doi: $paper_metadata.doi})
                    
                    WITH p
                    UNWIND $precursors as precursor_name
                    MERGE (pre:Precursor {name: precursor_name})
                    MERGE (p)-[:USED_PRECURSOR]->(pre)
                    """

                    session.run(precursors_query, **params)

                # Handle characterization techniques separately
                if synthesis_data.get("characterization_techniques"):
                    techniques_query = """
                    MATCH (p:Paper {doi: $paper_metadata.doi})
                    
                    WITH p
                    UNWIND $characterization_techniques as technique_name
                    MERGE (char:CharacterizationTechnique {name: technique_name})
                    MERGE (p)-[:USED_CHARACTERIZATION_TECHNIQUE]->(char)
                    """

                    session.run(techniques_query, **params)

                return True

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False

    def process_paper_data(
        self,
        synthesis_data: Dict,
        composition_data: Dict,
        paper_metadata: Dict,
    ) -> bool:
        """Process paper data and create nodes in Neo4j database"""
        try:
            success = self.create_paper_with_compositions(
                synthesis_data, composition_data, paper_metadata
            )
            if success:
                logging.info("Successfully created paper with authors and compositions")
            else:
                logging.error("Failed to create paper with authors and compositions")
            return success

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False
