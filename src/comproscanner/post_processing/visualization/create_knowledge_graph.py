"""
create_knowledge_graph.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 08-04-2025
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import json
from typing import Dict, Optional, List
from dotenv import load_dotenv
import os
from tqdm import tqdm
import sys
import difflib
from collections import Counter

# Try to import advanced NLP models for better semantic similarity
try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import utility functions
try:
    from ...utils.logger import setup_logger
    from ...utils.get_paper_data import PaperMetadataExtractor
    from ...utils.error_handler import ValueErrorHandler, ImportErrorHandler
except ImportError:
    print("Error importing utility modules. Make sure they are in the correct path.")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Setup logger
logger = setup_logger("post-processing.log")


class SemanticMatcher:
    """Class for semantic matching of text strings"""

    def __init__(self, model_name="thellert/physbert_cased"):
        """
        Initialize the semantic matching with the specified model

        Args:
            model_name (str): Name of the transformer model to load
        """
        self.semantic_model = None
        self._load_semantic_model(model_name)

    def _load_semantic_model(self, model_name):
        """
        Load the specified semantic model for similarity calculations.

        Args:
            model_name (str): Name of the model to load

        Returns:
            dict: Dictionary with model type and model/tokenizer objects
        """
        # Try loading the transformer model first
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Attempting to load {model_name} transformer model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self.semantic_model = {
                    "type": "transformers",
                    "tokenizer": tokenizer,
                    "model": model,
                }
                logger.info(f"Successfully loaded {model_name} transformer model")
                return self.semantic_model
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")

        # Try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Falling back to sentence-transformers model...")
                st_model = SentenceTransformer("all-mpnet-base-v2")
                self.semantic_model = {
                    "type": "sentence_transformer",
                    "model": st_model,
                }
                logger.info("Successfully loaded sentence-transformers model")
                return self.semantic_model
            except Exception as e:
                logger.warning(f"Could not load sentence-transformers: {e}")

        # Final fallback to difflib
        logger.info(
            "Falling back to difflib.SequenceMatcher for similarity calculations"
        )
        self.semantic_model = {"type": "difflib"}
        return self.semantic_model

    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using the best available method.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle empty or None inputs
        if not text1 or not text2:
            return 0.0

        # Convert to strings if needed
        text1 = str(text1)
        text2 = str(text2)

        # Use the appropriate similarity calculation method
        if self.semantic_model["type"] == "transformers":
            return self._calculate_similarity_transformers(text1, text2)
        elif self.semantic_model["type"] == "sentence_transformer":
            return self._calculate_similarity_sentence_transformer(text1, text2)
        else:
            # Fallback to difflib
            return self._calculate_similarity_difflib(text1, text2)

    def _calculate_similarity_transformers(self, text1, text2):
        """
        Calculate similarity between two texts using transformers model.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        tokenizer = self.semantic_model["tokenizer"]
        model = self.semantic_model["model"]

        # Tokenize and get embeddings
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # Use CLS token embedding (first token) as sentence representation
        emb1 = outputs1.last_hidden_state[:, 0, :]
        emb2 = outputs2.last_hidden_state[:, 0, :]

        # Normalize the embeddings
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Calculate cosine similarity
        similarity = torch.mm(emb1, emb2.transpose(0, 1)).item()

        return similarity

    def _calculate_similarity_sentence_transformer(self, text1, text2):
        """
        Calculate similarity between two texts using sentence-transformers.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        model = self.semantic_model["model"]

        # Get embeddings
        embedding1 = model.encode([text1])[0]
        embedding2 = model.encode([text2])[0]

        # Calculate cosine similarity
        import numpy as np

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Ensure result is in range [0, 1]
        return max(0.0, min(1.0, similarity))

    def _calculate_similarity_difflib(self, text1, text2):
        """
        Calculate similarity between two texts using difflib.SequenceMatcher.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def cluster_items(self, items, similarity_threshold=0.8):
        """
        Cluster similar items using semantic similarity.

        Args:
            items (list): List of items to cluster
            similarity_threshold (float): Minimum similarity to consider two items as same

        Returns:
            dict: Dictionary mapping canonical names to lists of similar items
        """
        if not items:
            return {}

        items_counter = Counter(items)
        sorted_items = sorted(items_counter.items(), key=lambda x: x[1], reverse=True)

        # Initialize clusters with the most frequent item as first canonical form
        clusters = {}
        processed = set()

        # Process items from most to least frequent
        for item, count in sorted_items:
            if item in processed:
                continue

            # Create a new cluster with this item as canonical form
            canonical = item
            clusters[canonical] = [item]
            processed.add(item)

            # Compare to all remaining items
            for other_item, other_count in sorted_items:
                if other_item in processed:
                    continue

                # Calculate similarity
                similarity = self.calculate_similarity(canonical, other_item)

                if similarity >= similarity_threshold:
                    clusters[canonical].append(other_item)
                    processed.add(other_item)

        return clusters

    def get_canonical_name(self, item, clusters):
        """
        Get the canonical name for an item from the clusters.

        Args:
            item (str): The item to look up
            clusters (dict): Dictionary mapping canonical names to lists of similar items

        Returns:
            str: The canonical name for the item, or the item itself if not found
        """
        for canonical, similar_items in clusters.items():
            if item in similar_items:
                return canonical
        return item


class CreateKG:
    def __init__(self):
        """Initialize Neo4j connection using environment variables"""
        self.driver = None
        self.semantic_matcher = SemanticMatcher()
        self.method_clusters = {}
        self.technique_clusters = {}
        self.keyword_clusters = {}

        try:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            database = os.getenv(
                "NEO4J_DATABASE", "neo4j"
            )  # Default to "neo4j" if not set

            if not all([uri, user, password]):
                raise ValueError(
                    "Missing required environment variables. Please check your .env file."
                )

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = database
            logger.info(f"Successfully connected to Neo4j database: {database}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except ValueError as e:
            logger.error(str(e))
            raise

    def close(self):
        """Close the driver connection"""
        if self.driver is not None:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_method_clusters(self, results):
        """
        Build clusters of semantically similar synthesis methods

        Args:
            results (dict): Dictionary containing results data

        Returns:
            dict: Dictionary mapping canonical methods to similar methods
        """
        # Extract all methods from results
        all_methods = []
        for _, paper_data in results.items():
            if (
                "synthesis_data" in paper_data
                and "method" in paper_data["synthesis_data"]
            ):
                method = paper_data["synthesis_data"]["method"]
                if method and method.strip().lower() != "unknown_method":
                    all_methods.append(method)

        # Cluster methods
        logger.info(f"Clustering {len(all_methods)} synthesis methods...")
        method_clusters = self.semantic_matcher.cluster_items(all_methods, 0.9)
        num_canonical = len(method_clusters)
        num_total = len(all_methods)
        logger.info(
            f"Clustered {num_total} synthesis methods into {num_canonical} canonical methods"
        )

        # Log some clusters as examples
        for i, (canonical, similar) in enumerate(method_clusters.items()):
            if (
                len(similar) > 1 and i < 5
            ):  # Show first 5 non-trivial clusters as examples
                logger.info(f"Method cluster example: '{canonical}' includes {similar}")

        return method_clusters

    def build_technique_clusters(self, results):
        """
        Build clusters of semantically similar characterization techniques

        Args:
            results (dict): Dictionary containing results data

        Returns:
            dict: Dictionary mapping canonical techniques to similar techniques
        """
        # Extract all techniques from results
        all_techniques = []
        for _, paper_data in results.items():
            if (
                "synthesis_data" in paper_data
                and "characterization_techniques" in paper_data["synthesis_data"]
            ):
                techniques = paper_data["synthesis_data"]["characterization_techniques"]
                if techniques:
                    all_techniques.extend(techniques)

        # Cluster techniques
        logger.info(f"Clustering {len(all_techniques)} characterization techniques...")
        technique_clusters = self.semantic_matcher.cluster_items(all_techniques, 0.9)
        num_canonical = len(technique_clusters)
        num_total = len(all_techniques)
        logger.info(
            f"Clustered {num_total} characterization techniques into {num_canonical} canonical techniques"
        )

        # Log some clusters as examples
        for i, (canonical, similar) in enumerate(technique_clusters.items()):
            if (
                len(similar) > 1 and i < 5
            ):  # Show first 5 non-trivial clusters as examples
                logger.info(
                    f"Technique cluster example: '{canonical}' includes {similar}"
                )

        return technique_clusters

    def build_keyword_clusters(self, results):
        """
        Build clusters of semantically similar keywords

        Args:
            results (dict): Dictionary containing results data

        Returns:
            dict: Dictionary mapping canonical keywords to similar keywords
        """
        # Extract all keywords from results
        all_keywords = []
        for _, paper_data in results.items():
            if (
                "article_metadata" in paper_data
                and "keywords" in paper_data["article_metadata"]
            ):
                keywords = paper_data["article_metadata"]["keywords"]
                if keywords:
                    all_keywords.extend(keywords)

        # Cluster keywords
        logger.info(f"Clustering {len(all_keywords)} keywords...")
        keyword_clusters = self.semantic_matcher.cluster_items(
            all_keywords, 0.85
        )  # Higher threshold for keywords
        num_canonical = len(keyword_clusters)
        num_total = len(all_keywords)
        logger.info(
            f"Clustered {num_total} keywords into {num_canonical} canonical keywords"
        )

        # Log some clusters as examples
        for i, (canonical, similar) in enumerate(keyword_clusters.items()):
            if (
                len(similar) > 1 and i < 5
            ):  # Show first 5 non-trivial clusters as examples
                logger.info(
                    f"Keyword cluster example: '{canonical}' includes {similar}"
                )

        return keyword_clusters

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

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Use empty strings instead of None for properties used in MERGE
                family_name = composition_data.get("family", "UNKNOWN_FAMILY")

                # Get original method name and use semantic matcher to find canonical name
                original_method_name = synthesis_data.get("method", "UNKNOWN_METHOD")
                method_name = self.semantic_matcher.get_canonical_name(
                    original_method_name, self.method_clusters
                )

                if original_method_name != method_name:
                    logger.info(
                        f"Mapped method '{original_method_name}' to canonical '{method_name}'"
                    )

                synthesis_steps = (
                    "\n".join(f"- {step}" for step in synthesis_data.get("steps", []))
                    or "No steps available"
                )

                # Verify if paper already exists
                check_query = "MATCH (p:Paper {doi: $doi}) RETURN p"
                result = session.run(check_query, doi=paper_metadata.get("doi"))
                exists = result.single() is not None

                if exists:
                    logger.warning(
                        f"Paper with DOI {paper_metadata.get('doi')} already exists in database. Skipping."
                    )
                    return True  # Consider it a success if it already exists

                # Process characterization techniques to use canonical names
                original_techniques = synthesis_data.get(
                    "characterization_techniques", []
                )
                canonical_techniques = []

                for technique in original_techniques:
                    canonical = self.semantic_matcher.get_canonical_name(
                        technique, self.technique_clusters
                    )
                    canonical_techniques.append(canonical)
                    if canonical != technique:
                        logger.debug(
                            f"Mapped technique '{technique}' to canonical '{canonical}'"
                        )

                # Process keywords to use canonical names
                original_keywords = paper_metadata.get("keywords", [])
                canonical_keywords = []

                for keyword in original_keywords:
                    canonical = self.semantic_matcher.get_canonical_name(
                        keyword, self.keyword_clusters
                    )
                    canonical_keywords.append(canonical)
                    if canonical != keyword:
                        logger.debug(
                            f"Mapped keyword '{keyword}' to canonical '{canonical}'"
                        )

                # Update with canonical technique names
                synthesis_data_modified = dict(synthesis_data)
                synthesis_data_modified["characterization_techniques"] = (
                    canonical_techniques
                )

                # Update with canonical keyword names
                paper_metadata_modified = dict(paper_metadata)
                paper_metadata_modified["keywords"] = canonical_keywords

                params = {
                    "synthesis_data": synthesis_data_modified or {},
                    "composition_data": composition_data or {},
                    "paper_metadata": paper_metadata_modified,
                    "synthesis_steps": synthesis_steps,
                    "family_name": family_name,
                    "method_name": method_name,
                    "compositions": composition_data.get(
                        "compositions_property_values", {}
                    ),
                    "property_unit": composition_data.get("property_unit"),
                    "precursors": synthesis_data.get("precursors", []),
                    "characterization_techniques": canonical_techniques,
                    "keywords": canonical_keywords,
                }

                # Add a transaction wrapper to ensure all-or-nothing operations
                tx = session.begin_transaction()
                try:
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

                    result = tx.run(query, **params)
                    paper_node = result.single()

                    if not paper_node:
                        logger.error(
                            f"Failed to create paper node for DOI: {paper_metadata.get('doi')}"
                        )
                        tx.rollback()
                        return False

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

                        tx.run(compositions_query, **params)

                    # Handle precursors separately
                    if synthesis_data.get("precursors"):
                        precursors_query = """
                        MATCH (p:Paper {doi: $paper_metadata.doi})
                        
                        WITH p
                        UNWIND $precursors as precursor_name
                        MERGE (pre:Precursor {name: precursor_name})
                        MERGE (p)-[:USED_PRECURSOR]->(pre)
                        """

                        tx.run(precursors_query, **params)

                    # Handle characterization techniques separately
                    if canonical_techniques:
                        techniques_query = """
                        MATCH (p:Paper {doi: $paper_metadata.doi})
                        
                        WITH p
                        UNWIND $characterization_techniques as technique_name
                        MERGE (char:CharacterizationTechnique {name: technique_name})
                        MERGE (p)-[:USED_CHARACTERIZATION_TECHNIQUE]->(char)
                        """

                        tx.run(techniques_query, **params)

                    # Handle keywords separately
                    if canonical_keywords:
                        keywords_query = """
                        MATCH (p:Paper {doi: $paper_metadata.doi})
                        
                        WITH p
                        UNWIND $keywords as keyword_name
                        MERGE (k:Keyword {name: keyword_name})
                        MERGE (p)-[:HAS_KEYWORD]->(k)
                        """

                        tx.run(keywords_query, **params)

                    # Commit the transaction
                    tx.commit()

                    # Verify the paper was actually created
                    verify_query = "MATCH (p:Paper {doi: $doi}) RETURN p"
                    result = session.run(verify_query, doi=paper_metadata.get("doi"))
                    verified = result.single() is not None

                    if verified:
                        logger.info(
                            f"VERIFIED: Paper node for DOI: {paper_metadata.get('doi')} exists in database."
                        )
                        return True
                    else:
                        logger.error(
                            f"VERIFICATION FAILED: Paper node for DOI: {paper_metadata.get('doi')} was not found after creation."
                        )
                        return False

                except Exception as e:
                    # If any error occurs, roll back the transaction
                    logger.error(
                        f"Transaction error for DOI {paper_metadata.get('doi')}: {e}"
                    )
                    if tx.closed() == False:
                        tx.rollback()
                    return False

                return True

        except Exception as e:
            logger.error(f"An error occurred: {e}")
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
                logger.info(
                    f"Successfully created paper with authors and compositions for DOI: {paper_metadata.get('doi')}"
                )
                return success
            else:
                logger.error(
                    f"Failed to create paper with authors and compositions for DOI: {paper_metadata.get('doi')}"
                )
                return False
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return False

    def create_knowledge_graph(self, result_file: str = "extracted_results.json"):
        """
        Create a knowledge graph from the extracted results file

        Args:
            result_file (str, required): Path to the JSON file containing extracted results.
        """

        # Define a local function to load results file
        def load_results_file(file_path: str) -> Dict:
            """
            Load extracted results from a JSON file

            Args:
                file_path: Path to the results JSON file

            Returns:
                Dict: Dictionary containing the extracted results
            """
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading results file {file_path}: {e}")
                raise

        try:
            # Load the extracted results
            logger.info(f"Loading results from {result_file}")
            results = load_results_file(result_file)

            # Build semantic clusters for methods, techniques, and keywords first
            self.method_clusters = self.build_method_clusters(results)
            self.technique_clusters = self.build_technique_clusters(results)
            self.keyword_clusters = self.build_keyword_clusters(results)

            # Initialize the paper metadata extractor
            paper_metadata_extractor = PaperMetadataExtractor()

            # Process each paper in the results
            logger.info(f"Processing {len(results)} papers")

            success_count = 0
            skipped_count = 0
            failed_count = 0

            for doi, paper_data in tqdm(
                results.items(), desc="Building knowledge graph"
            ):
                # Extract composition and synthesis data
                composition_data = paper_data.get("composition_data", {})
                synthesis_data = paper_data.get("synthesis_data", {})

                # Try to get paper metadata from existing data or fetch if not available
                paper_metadata = paper_data.get("article_metadata", {})

                if not paper_metadata:
                    try:
                        # Try to get paper metadata from external source
                        paper_metadata = paper_metadata_extractor.get_article_metadata(
                            doi
                        )
                    except Exception as e:
                        logger.error(
                            f"Error fetching paper metadata for DOI: {doi}. {e}"
                        )
                        # Use minimal metadata with just the DOI if fetching fails
                        paper_metadata = {"doi": doi}

                # Make sure a DOI is there in the metadata
                if "doi" not in paper_metadata:
                    paper_metadata["doi"] = doi

                # Validate required data
                if not paper_metadata.get("doi"):
                    logger.error(f"Missing DOI for paper. Skipping.")
                    failed_count += 1
                    continue

                # Process the paper data
                success = self.process_paper_data(
                    synthesis_data, composition_data, paper_metadata
                )

                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Failed to process paper with DOI: {doi}")

            # Summary stats
            total_expected = len(results)
            logger.info(f"Knowledge graph building process completed:")
            logger.info(f"  Total papers in input file: {total_expected}")
            logger.info(f"  Successfully processed: {success_count}")
            logger.info(f"  Failed: {failed_count}")

            # Verify actual count in database
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
                db_paper_count = result.single()["paper_count"]
                logger.info(f"  Total papers in database: {db_paper_count}")

                # Count unique methods after semantic clustering
                result = session.run("MATCH (m:Method) RETURN count(m) as method_count")
                method_count = result.single()["method_count"]
                logger.info(f"  Total unique methods in database: {method_count}")

                # Count unique characterization techniques after semantic clustering
                result = session.run(
                    "MATCH (c:CharacterizationTechnique) RETURN count(c) as technique_count"
                )
                technique_count = result.single()["technique_count"]
                logger.info(
                    f"  Total unique characterization techniques in database: {technique_count}"
                )

                # Count unique keywords after semantic clustering
                result = session.run(
                    "MATCH (k:Keyword) RETURN count(k) as keyword_count"
                )
                keyword_count = result.single()["keyword_count"]
                logger.info(f"  Total unique keywords in database: {keyword_count}")

            if db_paper_count != success_count:
                logger.warning(
                    f"Discrepancy detected: {success_count} papers reported as successful but only {db_paper_count} found in database"
                )

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
