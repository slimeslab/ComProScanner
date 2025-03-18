import os
from ..error_handler import ValueErrorHandler


class DatabaseConfig:
    def __init__(self, main_property_keyword, is_sql_db=None):
        self.PAPERDATA_TABLE_NAME = f"{main_property_keyword}_data"
        self.EXTRACTED_CSV_FOLDERPATH = (
            f"results/extracted_data/{main_property_keyword}"
        )
        if is_sql_db:
            db_user = os.getenv("DATABASE_USER")
            db_password = os.getenv("DATABASE_PASSWORD")
            db_host = os.getenv("DATABASE_HOST")
            db_name = os.getenv("DATABASE_NAME")

            missing_vars = []
            if db_user is None:
                missing_vars.append("DATABASE_USER")
            if db_password is None:
                missing_vars.append("DATABASE_PASSWORD")
            if db_host is None:
                missing_vars.append("DATABASE_HOST")
            if db_name is None:
                missing_vars.append("DATABASE_NAME")

            if missing_vars:
                raise ValueErrorHandler(
                    message=f"Required environment variables cannot be None. Missing variables: {', '.join(missing_vars)}"
                )

            self.DATABASE_CONNECTION_URL = (
                f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
            )
