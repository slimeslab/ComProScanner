"""
prepare_iop_files.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 10-03-2025
"""

# Standard library imports
import os
import shutil

# Third party imports
from lxml import etree
import pandas as pd
from tqdm import tqdm

# Custom imports
from .configs import DefaultPaths
from .error_handler import ValueErrorHandler, KeyboardInterruptHandler
from .common_functions import return_error_message


class PrepareIOPFiles:
    def __init__(self, main_property_keyword: str = None, logger=None):
        """
        Class to get the required IOP file paths and match the DOIs with the DataFrame.

        Args:
            main_property_keyword (str): The main property keyword to process the articles.
            logger (Logger): The logger object to log messages

        Raises:
            ValueErrorHandler: If the main_property_keyword is not provided
        """
        self.keyword = main_property_keyword
        if self.keyword is None:
            raise ValueErrorHandler(return_error_message("main_property_keyword"))
        self.logger = logger
        if self.logger is None:
            raise ValueErrorHandler(f"Logger not provided. Logger is required.")
        self.all_paths = DefaultPaths(self.keyword)
        self.metadata_csv_filename = self.all_paths.METADATA_CSV_FILENAME
        self.xml_folderpath = self.all_paths.IOP_FOLDERPATH
        self.df = pd.read_csv(self.metadata_csv_filename)

    def _return_xml_root(self, source_path: str) -> etree._Element:
        """
        Return the root of the XML file.

        Args:
            source_path (str): The path to the XML file

        Returns:
            etree._Element: The root of the XML file

        Raises:
            Exception: If there is an error parsing the XML file
        """
        try:
            tree = etree.parse(source_path.strip())
            return tree.getroot()
        except Exception as e:
            self.logger.error(f"Error parsing XML file {source_path}: {e}")
            return None

    def _get_modified_doi(self, source_path: str):
        """
        Get the modified DOI from the XML file.

        Args:
            source_path (str): The path to the XML file

        Returns:
            str: The modified DOI

        Raises:
            Exception: If there is an error parsing the XML file
        """
        try:
            root = self._return_xml_root(source_path)
            if root is not None:
                doi = root.xpath('.//*[local-name()="article-id"][@pub-id-type="doi"]')
                return doi[0].text.replace("/", "_") if doi else None
        except Exception as e:
            self.logger.error(f"Error getting the modified DOI from {source_path}: {e}")
            return None

    def _check_body_presence(self, source_path: str) -> bool:
        """
        Check if the body tag is present in the XML.

        Args:
            source_path (str): The path to the XML file

        Returns:
            bool: True if the body tag is present, False otherwise
        """
        root = self._return_xml_root(source_path)
        if root is None:
            return False
        body = root.xpath('.//*[local-name()="body"]')
        return True if body else False

    def _get_xml_files(self, folder_path: str) -> list:
        """
        Get list of all XML files in the folder and subfolders.

        Args:
            folder_path (str): Path to the folder to search

        Returns:
            list: List of tuples containing (root_path, filename)
        """
        xml_files = []
        parent_files = set(
            f for f in os.listdir(folder_path) if f.lower().endswith(".xml")
        )

        for root, _, files in os.walk(folder_path):
            if root == folder_path:  # Skip the parent directory
                continue
            for file in files:
                if file.lower().endswith(".xml"):
                    # Get DOI for the current file
                    source_path = os.path.join(root, file)
                    modified_doi = self._get_modified_doi(source_path)

                    if modified_doi:
                        new_name = f"{modified_doi}.xml"
                        # Only add if the file with this DOI doesn't exist in parent
                        if new_name not in parent_files:
                            xml_files.append((root, file))
                    else:
                        # If we can't get DOI, check if filename exists in parent
                        if file not in parent_files:
                            xml_files.append((root, file))

        return xml_files

    def _combine_xml_files(self, folder_path):
        """
        Move all XML files from subfolders to the main folder, rename them with DOI numbers,
        and remove empty folders.

        Args:
            folder_path (str): Path to the main folder
            doi_mapping (dict): Dictionary mapping original filenames to DOI numbers

        Returns:
            tuple: (number of files moved, number of folders deleted)
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            self.logger.error(f"Folder path does not exist: {folder_path}")
            raise ValueErrorHandler(f"Folder path does not exist: {folder_path}")

        xml_files = self._get_xml_files(folder_path)
        if not xml_files:
            self.logger.debug("All XML files are already in the parent directory")
            return 0

        files_moved = 0
        for root, file in tqdm(
            reversed(xml_files), total=len(xml_files), desc="Processing XML files"
        ):
            try:
                source_path = os.path.join(root, file)

                # Get the new name for the file from modified DOI
                modified_doi = self._get_modified_doi(source_path)
                if modified_doi:
                    new_name = f"{modified_doi}.xml"
                else:
                    self.logger.warning(f"Could not get modified DOI for {source_path}")
                    new_name = file

                if self._check_body_presence(source_path):
                    dest_path = os.path.join(folder_path, new_name)
                    if not os.path.exists(dest_path):
                        shutil.move(source_path, dest_path)
                        files_moved += 1
                else:
                    os.remove(source_path)
            except KeyboardInterrupt as kie:
                self.logger.error(f"Keyboard Interruption occured: {kie}")
                raise KeyboardInterruptHandler(kie)
            except Exception as e:
                self.logger.error(f"Error moving file {file}: {e}")

        # Remove empty folders
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError as e:
                    self.logger.error(f"Error removing directory {dir_path}: {e}")
        return files_moved

    def _process_zip_files(self, folder_path: str) -> int:
        """
        Process all ZIP files in the folder and subfolders to extract XML files with body tags and move them to the main folder with modified DOI names.

        Args:
            folder_path (str): Path to the main folder to search for ZIP files

        Returns:
            int: Number of XML files processed from ZIP archives
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            self.logger.error(f"Folder path does not exist: {folder_path}")
            raise ValueErrorHandler(f"Folder path does not exist: {folder_path}")

        # Find all ZIP files in the folder and subfolders
        zip_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_files.append(os.path.join(root, file))

        if not zip_files:
            self.logger.debug("No ZIP files found")
            return 0

        self.logger.info(f"Found {len(zip_files)} ZIP files to process")

        # Create temporary extraction directory
        temp_dir = os.path.join(folder_path, "temp_extraction")
        os.makedirs(temp_dir, exist_ok=True)

        files_processed = 0

        # Process each ZIP file
        for zip_file in tqdm(zip_files, desc="Processing ZIP files"):
            try:
                # Create a unique subfolder for each ZIP file
                zip_basename = os.path.basename(zip_file).replace(".zip", "")
                extract_dir = os.path.join(temp_dir, zip_basename)
                os.makedirs(extract_dir, exist_ok=True)
                self.logger.debug(f"Extracting {zip_file} to {extract_dir}")
                shutil.unpack_archive(zip_file, extract_dir)
                os.remove(zip_file)

                # Process extracted XML files
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith(".xml"):
                            source_path = os.path.join(root, file)
                            has_body = self._check_body_presence(source_path)
                            modified_doi = self._get_modified_doi(source_path)
                            if has_body and modified_doi:
                                new_name = f"{modified_doi}.xml"
                                dest_path = os.path.join(folder_path, new_name)
                                if not os.path.exists(dest_path):
                                    shutil.move(source_path, dest_path)
                                    files_processed += 1
                                else:
                                    self.logger.debug(
                                        f"File {new_name} already exists in destination"
                                    )
                                    os.remove(source_path)  # Remove duplicate
                            else:
                                os.remove(source_path)

            except KeyboardInterrupt as kie:
                self.logger.error(f"Keyboard Interruption occurred: {kie}")
                raise KeyboardInterruptHandler(kie)
            except Exception as e:
                self.logger.error(f"Error processing ZIP file {zip_file}: {e}")

        # Clean up temporary directories
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            self.logger.error(f"Error removing temporary directory {temp_dir}: {e}")

        return files_processed

    def prepare_files(self):
        """
        Main function to prepare the userful IOP files in one folder.
            1. Process ZIP files - extract, filter and move valid XML files
            2. Combine all XML files from subfolders to the main folder
        """
        self.logger.verbose("IOP files preparation started...")

        # Process all ZIP files
        self.logger.debug("Processing ZIP archives...")
        zip_files_processed = self._process_zip_files(self.xml_folderpath)
        self.logger.info(
            f"Processed {zip_files_processed} XML files from ZIP archives."
        )

        # Combine all XML files from subfolders
        self.logger.debug("Processing XML files in subfolders...")
        files_moved = self._combine_xml_files(self.xml_folderpath)
        self.logger.info(
            f"Moved {files_moved} XML files from subfolders to the main folder."
        )

        total_files_processed = zip_files_processed + files_moved
        self.logger.info(f"Total files processed: {total_files_processed}")
        self.logger.verbose("IOP files preparation completed.\n")

    def get_all_iop_dois(self) -> list:
        """
        Return all available IOP paper dois in the main folder.

        Returns:
            list: List of all available IOP paper DOIs in the main folder
        """
        iop_dois = []
        for file in os.listdir(self.xml_folderpath):
            if file.lower().endswith(".xml"):
                # Remove .xml extension and replace underscores with forward slashes
                base_name = file[:-4]
                modified_name = base_name.replace("_", "/")
                iop_dois.append(modified_name)
        return iop_dois
