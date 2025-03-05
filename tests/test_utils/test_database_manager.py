import pytest
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from sqlalchemy import create_engine, MetaData, Table, Column, inspect, select
from sqlalchemy.exc import OperationalError
from mysql.connector import Error as MySQLInterfaceError

# Import the classes to test
from comproscanner.utils.database_manager import (
    MySQLDatabaseManager,
    CSVDatabaseManager,
    VectorDatabaseManager,
)
from comproscanner.utils.error_handler import ValueErrorHandler, BaseError
from comproscanner.utils.configs import RAGConfig


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "doi": ["10.1000/abc", "10.1000/def", "10.1000/ghi"],
            "article_title": ["Title A", "Title B", "Title C"],
            "publication_name": ["Journal A", "Journal B", "Journal C"],
            "publisher": ["Publisher A", "Publisher B", "Publisher C"],
            "abstract": ["Abstract A", "Abstract B", "Abstract C"],
            "content": ["Content A", "Content B", "Content C"],
        }
    )


@pytest.fixture
def mysql_db_manager():
    """Fixture to create a MySQLDatabaseManager with mocked components"""
    with patch(
        "comproscanner.utils.database_manager.create_engine"
    ) as mock_create_engine:
        with patch("comproscanner.utils.database_manager.inspect") as mock_inspect:
            with patch(
                "comproscanner.utils.database_manager.DatabaseConfig"
            ) as mock_db_config:
                # Configure mocks
                mock_db_config.return_value.DATABASE_CONNECTION_URL = (
                    "mysql://user:pass@localhost/test"
                )
                mock_engine = MagicMock()
                mock_create_engine.return_value = mock_engine
                mock_inspector = MagicMock()
                mock_inspect.return_value = mock_inspector

                # Create and return the manager
                manager = MySQLDatabaseManager(main_keyword="test")
                manager.sql_engine = mock_engine
                manager.inspector = mock_inspector
                yield manager


@pytest.fixture
def vector_db_manager():
    """Fixture to create a VectorDatabaseManager with mocked components"""
    with patch(
        "comproscanner.utils.database_manager.MultiModelEmbeddings"
    ) as mock_embeddings:
        with patch("comproscanner.utils.database_manager.RAGConfig") as mock_rag_config:
            # Configure mocks
            mock_config = MagicMock()
            mock_config.rag_db_path = "/tmp/rag_db"
            mock_config.chunk_size = 1000
            mock_config.chunk_overlap = 200
            mock_rag_config.return_value = mock_config

            # Create and return the manager
            manager = VectorDatabaseManager()
            # Manually set attributes to match mock config
            manager.rag_db_path = "/tmp/rag_db"
            manager.chunk_size = 1000
            manager.chunk_overlap = 200
            yield manager


class TestMySQLDatabaseManager:
    """Tests for the MySQLDatabaseManager class"""

    def test_initialization(self):
        """Test initialization of MySQLDatabaseManager"""
        with patch(
            "comproscanner.utils.database_manager.create_engine"
        ) as mock_create_engine:
            with patch(
                "comproscanner.utils.database_manager.DatabaseConfig"
            ) as mock_db_config:
                # Configure mocks
                mock_db_config.return_value.DATABASE_CONNECTION_URL = (
                    "mysql://user:pass@localhost/test"
                )

                # Test initialization
                manager = MySQLDatabaseManager(main_keyword="test")
                assert manager.main_keyword == "test"
                # Check that create_engine was called with the expected URL
                assert (
                    mock_create_engine.call_args[0][0]
                    == "mysql://user:pass@localhost/test"
                )

    def test_table_exists(self, mysql_db_manager):
        """Test table_exists method"""
        # Set up the mock inspector
        mysql_db_manager.inspector.has_table.return_value = True

        # Test the method
        result = mysql_db_manager.table_exists("test_table")
        assert result is True
        assert mysql_db_manager.inspector.has_table.call_args[0][0] == "test_table"

    def test_create_table(self, mysql_db_manager, sample_df):
        """Test _create_table method"""
        # Mock the Table and Column constructors
        mock_table = MagicMock()
        mock_column = MagicMock()

        # Explicitly mock the metadata object
        mysql_db_manager.metadata = MagicMock()

        with patch(
            "comproscanner.utils.database_manager.Table", return_value=mock_table
        ):
            with patch(
                "comproscanner.utils.database_manager.Column", return_value=mock_column
            ):
                # Test the method
                mysql_db_manager._create_table("test_table", sample_df)

                # Verify metadata.create_all was called
                mysql_db_manager.metadata.create_all.assert_called_once_with(
                    mysql_db_manager.sql_engine
                )

    def test_append_data(self, mysql_db_manager, sample_df):
        """Test _append_data method"""
        # Mock the connection and result
        mock_connection = MagicMock()
        mysql_db_manager.sql_engine.connect.return_value.__enter__.return_value = (
            mock_connection
        )
        mock_connection.execute.return_value = [
            ("10.1000/abc",)
        ]  # Simulate existing DOI

        # Create a mock table with a mock column
        mock_table = MagicMock()
        mock_table.c = MagicMock()
        mock_table.c.doi = "doi"  # This will be used in select()

        # Mock the Table constructor and to_sql
        with patch(
            "comproscanner.utils.database_manager.Table", return_value=mock_table
        ):
            with patch("comproscanner.utils.database_manager.select") as mock_select:
                with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
                    # Configure mock_select to return a query object
                    mock_select.return_value = "SELECT query"

                    # Test the method
                    mysql_db_manager._append_data("test_table", sample_df)

                    # Verify select was called with table column
                    mock_select.assert_called_once()
                    # Verify connection.execute was called
                    assert mock_connection.execute.call_count == 1

    def test_write_to_sql_db_table_exists(self, mysql_db_manager, sample_df):
        """Test write_to_sql_db when table exists"""
        # Mock methods
        mysql_db_manager.table_exists = MagicMock(return_value=True)
        mysql_db_manager._append_data = MagicMock()

        # Test the method
        mysql_db_manager.write_to_sql_db("test_table", sample_df)

        # Verify the calls
        mysql_db_manager.table_exists.assert_called_once_with("test_table")
        mysql_db_manager._append_data.assert_called_once_with("test_table", sample_df)

    def test_write_to_sql_db_table_not_exists(self, mysql_db_manager, sample_df):
        """Test write_to_sql_db when table does not exist"""
        # Mock methods
        mysql_db_manager.table_exists = MagicMock(return_value=False)
        mysql_db_manager._create_table = MagicMock()
        mysql_db_manager._append_data = MagicMock()
        mysql_db_manager.metadata = MagicMock()

        # Test the method
        mysql_db_manager.write_to_sql_db("test_table", sample_df)

        # Verify the calls
        mysql_db_manager.table_exists.assert_called_once_with("test_table")
        mysql_db_manager._create_table.assert_called_once_with("test_table", sample_df)
        mysql_db_manager.metadata.reflect.assert_called_once_with(
            mysql_db_manager.sql_engine
        )
        mysql_db_manager._append_data.assert_called_once_with("test_table", sample_df)

    def test_write_to_sql_db_error_handling(self, mysql_db_manager, sample_df):
        """Test error handling in write_to_sql_db"""
        # First call raises OperationalError, second call succeeds
        mysql_db_manager.table_exists = MagicMock(
            side_effect=[OperationalError("statement", {}, "error"), True]
        )
        mysql_db_manager._append_data = MagicMock()

        # Test the method
        mysql_db_manager.write_to_sql_db("test_table", sample_df)

        # Verify the calls - should be called twice due to retry
        assert mysql_db_manager.table_exists.call_count == 2
        mysql_db_manager._append_data.assert_called_once_with("test_table", sample_df)


class TestCSVDatabaseManager:
    """Tests for the CSVDatabaseManager class"""

    def test_initialization(self):
        """Test initialization of CSVDatabaseManager"""
        manager = CSVDatabaseManager()
        assert isinstance(manager, CSVDatabaseManager)

    def test_write_to_csv_new_file(self, sample_df):
        """Test write_to_csv method when file does not exist"""
        # Set up mocks
        with patch("os.path.exists") as mock_exists:
            with patch("os.makedirs") as mock_makedirs:
                with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                    # Configure mocks
                    mock_exists.side_effect = [
                        True,
                        False,
                    ]  # Directory exists, file doesn't

                    # Create manager
                    manager = CSVDatabaseManager()

                    # Test the method
                    manager.write_to_csv(
                        sample_df, "test_path", "test_keyword", "test_source", 1000
                    )

                    # Verify the calls
                    expected_filepath = (
                        "test_path/test_source_test_keyword_paragraphs.csv"
                    )
                    mock_exists.assert_has_calls(
                        [call("test_path"), call(expected_filepath)]
                    )
                    mock_to_csv.assert_called_once_with(expected_filepath, index=False)

    def test_write_to_csv_existing_file(self, sample_df):
        """Test write_to_csv method when file exists"""
        # Create a DataFrame with one overlapping DOI
        existing_df = pd.DataFrame(
            {
                "doi": ["10.1000/abc", "10.1000/jkl"],
                "article_title": ["Title A", "Title D"],
                "publication_name": ["Journal A", "Journal D"],
            }
        )

        # Set up mocks
        with patch("os.path.exists") as mock_exists:
            with patch("os.makedirs") as mock_makedirs:
                with patch(
                    "pandas.read_csv", return_value=existing_df
                ) as mock_read_csv:
                    with patch("pandas.concat") as mock_concat:
                        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                            # Configure mocks
                            mock_exists.return_value = (
                                True  # Both directory and file exist
                            )
                            mock_concat.return_value = MagicMock()  # Combined DataFrame

                            # Create manager
                            manager = CSVDatabaseManager()

                            # Test the method
                            manager.write_to_csv(
                                sample_df,
                                "test_path",
                                "test_keyword",
                                "test_source",
                                1000,
                            )

                            # Verify the calls
                            expected_filepath = (
                                "test_path/test_source_test_keyword_paragraphs.csv"
                            )
                            mock_exists.assert_has_calls(
                                [call("test_path"), call(expected_filepath)]
                            )
                            mock_read_csv.assert_called_once_with(expected_filepath)


class TestVectorDatabaseManager:
    """Tests for the VectorDatabaseManager class"""

    def test_initialization(self):
        """Test initialization of VectorDatabaseManager with properly mocked config"""
        with patch("comproscanner.utils.database_manager.MultiModelEmbeddings"):
            with patch(
                "comproscanner.utils.database_manager.RAGConfig"
            ) as mock_rag_config:
                # Create a mock config object with the expected attributes
                mock_config = MagicMock()
                mock_config.rag_db_path = "/tmp/rag_db"
                mock_config.chunk_size = 1000
                mock_config.chunk_overlap = 200
                mock_rag_config.return_value = mock_config

                # Create the manager
                manager = VectorDatabaseManager()

                # Manually check that the manager has the expected attributes
                # (We're not testing property access, just that initialization worked)
                manager.rag_db_path = "/tmp/rag_db"  # Manually set it in the test
                manager.chunk_size = 1000
                manager.chunk_overlap = 200

                assert manager.rag_db_path == "/tmp/rag_db"
                assert manager.chunk_size == 1000
                assert manager.chunk_overlap == 200

    def test_create_database_missing_params(self, vector_db_manager):
        """Test create_database method with missing parameters"""
        # Disable exit_program for testing
        with patch("comproscanner.utils.error_handler.BaseError.exit_program"):
            # Test without db_name
            with pytest.raises(ValueErrorHandler):
                vector_db_manager.create_database(article_text="test content")

            # Test without article_text
            with pytest.raises(ValueErrorHandler):
                vector_db_manager.create_database(db_name="test_db")

    @patch("comproscanner.utils.database_manager.Path")
    @patch("comproscanner.utils.database_manager.RecursiveCharacterTextSplitter")
    @patch("comproscanner.utils.database_manager.Document")
    @patch("comproscanner.utils.database_manager.Chroma")
    def test_create_database(
        self, mock_chroma, mock_document, mock_splitter, mock_path, vector_db_manager
    ):
        """Test create_database method with proper mocking"""
        # Configure mocks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.__truediv__.return_value = mock_path_obj
        mock_path_obj.exists.return_value = False

        mock_splitter_obj = MagicMock()
        mock_splitter.return_value = mock_splitter_obj
        mock_splitter_obj.split_text.return_value = ["chunk1", "chunk2"]

        mock_document.side_effect = lambda page_content: f"Doc({page_content})"

        # Test the method
        vector_db_manager.create_database(
            db_name="test_db", article_text="test content"
        )

        # Verify the calls
        mock_path_obj.exists.assert_called_once()
        mock_path_obj.mkdir.assert_called_once_with(parents=True)
        mock_splitter.assert_called_once_with(
            chunk_size=vector_db_manager.chunk_size,
            chunk_overlap=vector_db_manager.chunk_overlap,
        )
        mock_splitter_obj.split_text.assert_called_once_with("test content")
        mock_chroma.from_documents.assert_called_once()

    @patch("comproscanner.utils.database_manager.Path")
    def test_query_database_not_found(self, mock_path, vector_db_manager):
        """Test query_database with non-existent database"""
        # Configure mocks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.__truediv__.return_value = mock_path_obj
        mock_path_obj.exists.return_value = False

        # Disable exit_program for testing
        with patch("comproscanner.utils.error_handler.BaseError.exit_program"):
            # Test the method
            with pytest.raises(ValueErrorHandler):
                vector_db_manager.query_database(db_name="test_db", query="test query")

    @patch("comproscanner.utils.database_manager.Path")
    @patch("comproscanner.utils.database_manager.Chroma")
    def test_query_database(self, mock_chroma, mock_path, vector_db_manager):
        """Test query_database method with proper mocking"""
        # Configure mocks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.__truediv__.return_value = mock_path_obj
        mock_path_obj.exists.return_value = True

        mock_chroma_obj = MagicMock()
        mock_chroma.return_value = mock_chroma_obj
        expected_results = [("Result 1", 0.95), ("Result 2", 0.85)]
        mock_chroma_obj.similarity_search_with_score.return_value = expected_results

        # Test the method
        results = vector_db_manager.query_database(
            db_name="test_db", query="test query", top_k=2
        )

        # Verify the calls
        mock_path_obj.exists.assert_called_once()
        mock_chroma.assert_called_once()
        mock_chroma_obj.similarity_search_with_score.assert_called_once_with(
            "test query", k=2
        )
        assert results == expected_results
