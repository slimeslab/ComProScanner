import os
from dotenv import load_dotenv
import requests

load_dotenv()

scopus_api_key = os.getenv("SCOPUS_API_KEY")


def get_elsevier_article(doi):
    """Fetch Elsevier article XML by DOI using the Scopus API."""
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {"X-ELS-APIKey": scopus_api_key, "Accept": "application/xml"}

    try:
        # Send GET request with headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the response content to a file
            folder_path = "Elsevier_xml_data"
            os.makedirs(folder_path, exist_ok=True)
            with open(
                os.path.join(folder_path, f"{doi.replace('/', '_')}.xml"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(response.text)
        else:
            print(f"Request failed with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    with open("../selected_dois.txt", "r", encoding="utf-8") as f:
        doi_list = [line.strip() for line in f.readlines() if line.strip()]

    for doi in doi_list:
        get_elsevier_article(doi)
        print(f"Article for DOI: {doi} fetched and saved.")
