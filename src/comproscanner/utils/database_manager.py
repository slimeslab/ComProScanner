"""
database_store.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-02-2025
"""

# Standard library imports
import os
import gc
from pathlib import Path
from typing import List

# Third-party imports
import pandas as pd
from sqlalchemy import (
    inspect,
    create_engine,
    select,
    MetaData,
    Table,
    Column,
    Integer,
    Text,
    String,
    exc as sqlalchemy_exc,
)
from mysql.connector import Error as MySQLInterfaceError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

# Custom imports
from .configs import DatabaseConfig, RAGConfig
from .error_handler import ValueErrorHandler
from .logger import setup_logger
from .embeddings import MultiModelEmbeddings

# configure logger
logger = setup_logger("comproscanner.log", module_name="database_manager")


class MySQLDatabaseManager:
    def __init__(self, main_keyword: str, is_sql_db: bool = False):
        if is_sql_db:
            try:
                is_sql_db = True
                self.main_keyword = main_keyword
                self.db_config = DatabaseConfig(self.main_keyword, is_sql_db)
                self.sql_db_url = self.db_config.DATABASE_CONNECTION_URL
                self.sql_engine = create_engine(self.sql_db_url)
                self.inspector = inspect(self.sql_engine)
                self.metadata = MetaData()
            except Exception as e:
                logger.error(f"Error: {e}...")

    def table_exists(self, table_name):
        try:
            return self.inspector.has_table(table_name)
        except Exception as e:
            logger.error(f"Error: {e}...")

    def _create_table(self, table_name, df):
        while True:
            try:
                table = Table(table_name, self.metadata)

                # Add an 'id' column as the first column
                id_column = Column("id", Integer, primary_key=True, autoincrement=True)
                table.append_column(id_column)

                # Add the DataFrame's columns
                for column in df.columns:
                    if column == "article_title":
                        new_column = Column(column, String(255))
                    elif column == "publication_name":
                        new_column = Column(column, String(125))
                    elif column in ["doi", "publisher"]:
                        new_column = Column(column, String(50))
                    else:
                        new_column = Column(column, Text)
                    table.append_column(new_column)

                # Create the table
                self.metadata.create_all(self.sql_engine)

                # Reconnect to the database
                self.sql_engine.dispose()
                self.sql_engine = create_engine(self.sql_db_url)

                break
            except MySQLInterfaceError:
                continue

    def _append_data(self, table_name, df):
        while True:
            try:
                table = Table(table_name, self.metadata, autoload_with=self.sql_engine)
                with self.sql_engine.connect() as connection:
                    stmt = select(table.c.doi)
                    result = connection.execute(stmt)
                    existing_dois = {row[0] for row in result}
                    # Filter the DataFrame
                    df = df[~df["doi"].isin(existing_dois)]

                    # Append the filtered DataFrame to the database
                    for _, row in df.iterrows():
                        row.to_frame().T.to_sql(
                            table_name,
                            self.sql_engine,
                            index=False,
                            index_label="id",
                            if_exists="append",
                        )
                break
            except MySQLInterfaceError:
                continue

    def write_to_sql_db(self, table_name, final_df):
        # If the new dataframe is not empty, append it to the database
        if not final_df.empty:
            while True:
                try:
                    logger.debug(f"\nWriting to database...\n")
                    if not self.table_exists(table_name):
                        logger.warning(f"Table does not exist...creating table...")
                        self._create_table(table_name, final_df)
                        self.metadata.reflect(self.sql_engine)

                    logger.info(f"Appending data to table...")
                    self._append_data(table_name, final_df)
                    break
                except sqlalchemy_exc.OperationalError:
                    logger.warning(f"Database connection failed...RETRYING...")
                    continue
                except MySQLInterfaceError:
                    logger.warning(
                        f"Lost connection to MySQL server during query...RETRYING..."
                    )
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}...")
                    break


class CSVDatabaseManager:
    def __init__(self):
        pass

    def write_to_csv(self, final_df, filepath, keyword, source, csv_batch_size):
        if csv_batch_size > 1:
            logger.info("Writing to CSV...")
        try:
            if not os.path.exists(filepath):
                os.makedirs(filepath)

            output_file = f"{filepath}/{source}_{keyword}_paragraphs.csv"

            if os.path.exists(output_file):
                # Read all columns as strings to avoid mixed type issues
                existing_df = pd.read_csv(output_file, dtype=str)
                final_df = final_df[~final_df["doi"].isin(existing_df["doi"])]
                if not final_df.empty:
                    combined_df = pd.concat([existing_df, final_df], ignore_index=True)
                    combined_df.to_csv(output_file, index=False)
            else:
                final_df.to_csv(output_file, index=False)

        except Exception as e:
            logger.error(f"Error: {e}")


class VectorDatabaseManager:
    """Leak-safe, auto-persisting ChromaDB handler for multiple vector databases."""

    def __init__(self, rag_config):
        self.rag_config = rag_config
        self.embeddings = MultiModelEmbeddings(rag_config)
        self.rag_db_path = Path(rag_config.rag_db_path)
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = rag_config.chunk_overlap
        self.client = PersistentClient(path=str(self.rag_db_path))

    def create_database(self, db_name: str, article_text: str):
        """Create a new persistent ChromaDB database (auto‑persisted)."""
        if not db_name:
            raise ValueError("Database name is required")
        if not article_text:
            raise ValueError("Article text is required")

        db_location = self.rag_db_path / db_name
        db_location.mkdir(parents=True, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = splitter.split_text(article_text)
        docs = [Document(page_content=t) for t in texts]

        # Automatic persistence of the vector database
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=str(db_location),
        )

        # Clear cache and release handles to avoid leaks
        self.client.clear_system_cache()
        del vectordb
        gc.collect()

        logger.info(f"Vector database auto-persisted at {db_location}")

    def query_database(self, db_name: str, query: str, top_k: int = 5):
        """Query the persisted ChromaDB database."""
        db_location = self.rag_db_path / db_name
        if not db_location.exists():
            raise ValueError(f"Database {db_name} not found at {db_location}")

        vectordb = Chroma(
            persist_directory=str(db_location),
            embedding_function=self.embeddings,
        )
        results = vectordb.similarity_search_with_score(query, k=top_k)

        # Clear cache and release handles to avoid leaks
        self.client.clear_system_cache()
        del vectordb
        gc.collect()

        logger.info(f"Retrieved {len(results)} results from {db_name}")
        return results

    def database_exists(self, db_name: str) -> bool:
        """Check if a vector database exists."""
        db_location = self.rag_db_path / db_name
        return db_location.exists()
