"""
base_urls.py - Contains the base URLs for the project

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""


class BaseUrls:
    METADATA_QUERY_BASE_URL = "https://api.elsevier.com/content/search/scopus?query="
    ISSN_BASE_URL = "https://api.elsevier.com/content/serial/title/issn/"
    SCOPUSID_BASE_URL = "https://api.elsevier.com/content/abstract/scopus_id/"
    UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2/"
    ELSEVIER_ARTICLE_BASE_URL = "https://api.elsevier.com/content/article/doi/"
    WILEY_ARTICLE_BASE_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
    SPRINGER_OPENACCESS_BASE_URL = (
        "https://api.springernature.com/openaccess/jats?q=doi:"
    )
    SPRINGER_TDM_BASE_URL = "https://spdi.public.springernature.app/xmldata/jats?q=doi:"
