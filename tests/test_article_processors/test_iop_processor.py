import os
import shutil
import pytest
from unittest.mock import Mock, patch, MagicMock
from lxml import etree
import pandas as pd
from io import StringIO
from comproscanner.utils.prepare_iop_files import PrepareIOPFiles
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    KeyboardInterruptHandler,
)


@pytest.fixture
def sample_xml_content():
    """Fixture to provide sample XML content"""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="doi">10.1088/sample_doi</article-id>
            </article-meta>
        </front>
        <body>
            <p>Sample content</p>
        </body>
    </article>
    """


@pytest.fixture
def sample_xml_without_body():
    """Fixture to provide sample XML content without body tag"""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="doi">10.1088/sample_doi_no_body</article-id>
            </article-meta>
        </front>
    </article>
    """


@pytest.fixture
def sample_metadata_csv():
    """Fixture to provide sample metadata CSV content"""
    data = """doi,publisher,article_type
10.1088/sample_doi,iop,Article
10.1088/another_doi,iop,Article
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def mock_logger():
    """Fixture to provide a mock logger"""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.verbose = Mock()
    return logger


@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    """Fixture to create temporary file structure and mock paths"""
    iop_dir = tmp_path / "iop_files"
    iop_dir.mkdir()
    subdir = iop_dir / "subdir"
    subdir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "doi": ["10.1088/sample_doi", "10.1088/another_doi"],
            "publisher": ["iop", "iop"],
            "article_type": ["Article", "Article"],
        }
    ).to_csv(csv_path, index=False)
    mock_default_paths = Mock()
    mock_default_paths.METADATA_CSV_FILENAME = str(csv_path)
    mock_default_paths.IOP_FOLDERPATH = str(iop_dir)

    with patch(
        "comproscanner.utils.prepare_iop_files.DefaultPaths",
        return_value=mock_default_paths,
    ):
        yield {"csv_path": csv_path, "iop_dir": iop_dir, "subdir": subdir}


def test_init_with_valid_params(mock_logger, mock_paths):
    """Test initialization with valid parameters"""
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    assert prep.keyword == "piezoelectric"
    assert prep.logger == mock_logger
    assert isinstance(prep.df, pd.DataFrame)


def test_init_without_keyword(mock_logger):
    """Test initialization without main property keyword"""
    with pytest.raises(ValueErrorHandler) as exc_info:
        PrepareIOPFiles(main_property_keyword=None, logger=mock_logger)
    assert "main_property_keyword" in str(exc_info.value)


def test_init_without_logger():
    """Test initialization without logger"""
    with pytest.raises(ValueErrorHandler) as exc_info:
        PrepareIOPFiles(main_property_keyword="piezoelectric", logger=None)
    assert "Logger not provided" in str(exc_info.value)


def test_return_xml_root(mock_logger, mock_paths, sample_xml_content, tmp_path):
    """Test _return_xml_root method with valid XML"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(sample_xml_content)
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    root = prep._return_xml_root(str(xml_path))
    assert root is not None
    assert root.tag == "article"
    assert len(root.xpath('.//*[local-name()="article-id"][@pub-id-type="doi"]')) == 1


def test_return_xml_root_invalid_file(mock_logger, mock_paths):
    """Test _return_xml_root method with invalid XML file"""
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    root = prep._return_xml_root("nonexistent_file.xml")
    assert root is None
    mock_logger.error.assert_called_once()


def test_get_modified_doi(mock_logger, mock_paths, sample_xml_content, tmp_path):
    """Test _get_modified_doi method"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(sample_xml_content)
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    modified_doi = prep._get_modified_doi(str(xml_path))
    assert modified_doi == "10.1088_sample_doi"


def test_get_modified_doi_no_doi(mock_logger, mock_paths, tmp_path):
    """Test _get_modified_doi method with XML missing DOI"""
    xml_path = tmp_path / "test_no_doi.xml"
    xml_path.write_text("""<?xml version="1.0" encoding="UTF-8"?><article></article>""")
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    modified_doi = prep._get_modified_doi(str(xml_path))
    assert modified_doi is None


def test_check_body_presence(
    mock_logger, mock_paths, sample_xml_content, sample_xml_without_body, tmp_path
):
    """Test _check_body_presence method"""
    xml_with_body = tmp_path / "with_body.xml"
    xml_with_body.write_text(sample_xml_content)
    xml_without_body = tmp_path / "without_body.xml"
    xml_without_body.write_text(sample_xml_without_body)
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    assert prep._check_body_presence(str(xml_with_body)) is True
    assert prep._check_body_presence(str(xml_without_body)) is False


@patch("os.listdir")
@patch("os.walk")
def test_get_xml_files(mock_walk, mock_listdir, mock_logger, mock_paths):
    """Test _get_xml_files method"""
    mock_listdir.return_value = ["existing.xml"]
    mock_walk.return_value = [
        (str(mock_paths["iop_dir"]), [], ["parent.xml"]),
        (str(mock_paths["subdir"]), [], ["file1.xml", "file2.xml"]),
    ]
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    prep._get_modified_doi = Mock(
        side_effect=lambda path: "10.1088_modified" if "file1" in path else None
    )
    xml_files = prep._get_xml_files(str(mock_paths["iop_dir"]))
    assert len(xml_files) == 2
    assert (str(mock_paths["subdir"]), "file1.xml") in xml_files
    assert (str(mock_paths["subdir"]), "file2.xml") in xml_files


@patch("os.path.exists")
@patch("shutil.move")
@patch("os.remove")
def test_combine_xml_files(
    mock_remove, mock_move, mock_exists, mock_logger, mock_paths
):
    """Test _combine_xml_files method"""
    mock_exists.return_value = True
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    prep._get_xml_files = Mock(return_value=[(str(mock_paths["subdir"]), "file1.xml")])
    prep._check_body_presence = Mock(return_value=True)  # Always return True
    prep._get_modified_doi = Mock(return_value="10.1088_modified")
    mock_exists.side_effect = lambda path: not path.endswith("10.1088_modified.xml")
    files_moved = prep._combine_xml_files(str(mock_paths["iop_dir"]))
    assert files_moved == 1
    mock_move.assert_called_once()


def test_process_zip_files(mock_logger, mock_paths):
    """Test _process_zip_files method"""
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    with patch("os.path.exists", return_value=True):
        with (
            patch("os.walk") as mock_walk,
            patch("os.makedirs") as mock_makedirs,
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_walk.return_value = [(str(mock_paths["iop_dir"]), [], [])]
            files_processed = prep._process_zip_files(str(mock_paths["iop_dir"]))
            assert files_processed == 0
            mock_makedirs.assert_not_called()
            mock_rmtree.assert_not_called()


def test_get_all_iop_dois(mock_logger, mock_paths):
    """Test get_all_iop_dois method"""
    with patch(
        "os.listdir",
        return_value=[
            "10.1088_sample_doi.xml",
            "10.1088_another_doi.xml",
            "README.txt",
        ],
    ):
        prep = PrepareIOPFiles(
            main_property_keyword="piezoelectric", logger=mock_logger
        )
        with patch.object(
            prep,
            "get_all_iop_dois",
            return_value=["10.1088/sample/doi", "10.1088/another/doi"],
        ):
            dois = prep.get_all_iop_dois()

            assert len(dois) == 2
            assert "10.1088/sample/doi" in dois
            assert "10.1088/another/doi" in dois


@patch("comproscanner.utils.prepare_iop_files.PrepareIOPFiles._process_zip_files")
@patch("comproscanner.utils.prepare_iop_files.PrepareIOPFiles._combine_xml_files")
def test_prepare_iop_files(mock_combine, mock_process_zip, mock_logger, mock_paths):
    """Test prepare_iop_files method"""
    mock_process_zip.return_value = 5
    mock_combine.return_value = 10
    prep = PrepareIOPFiles(main_property_keyword="piezoelectric", logger=mock_logger)
    prep.prepare_files()
    mock_process_zip.assert_called_once_with(prep.xml_folderpath)
    mock_combine.assert_called_once_with(prep.xml_folderpath)
    assert mock_logger.info.call_count >= 3
    assert mock_logger.verbose.call_count == 2


def test_keyboard_interrupt_in_process_zip_files(mock_logger, mock_paths):
    """Test KeyboardInterrupt handling in _process_zip_files method"""
    with (
        patch("os.path.exists", return_value=True),
        patch("os.walk", return_value=[("path", [], ["test.zip"])]),
        patch("os.makedirs"),
    ):
        prep = PrepareIOPFiles(
            main_property_keyword="piezoelectric", logger=mock_logger
        )
        with patch("shutil.unpack_archive", side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterruptHandler):
                prep._process_zip_files(str(mock_paths["iop_dir"]))
        mock_logger.error.assert_called_once()


def test_invalid_folder_path(mock_logger, mock_paths):
    """Test handling of invalid folder path"""
    with patch("os.path.exists", return_value=False):
        prep = PrepareIOPFiles(
            main_property_keyword="piezoelectric", logger=mock_logger
        )
        with pytest.raises(ValueErrorHandler):
            prep._combine_xml_files("invalid_path")
        with pytest.raises(ValueErrorHandler):
            prep._process_zip_files("invalid_path")
