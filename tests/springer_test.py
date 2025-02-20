import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Specify the URL
# url = f"http://api.springernature.com/openaccess/jats?q=doi:10.1007/s42114-024-00879-6&api_key={os.getenv('SPRINGER_OPENACCESS_API_KEY')}"  # Used for only open access articles

url = f"https://spdi.public.springernature.app/xmldata/jats?q=doi:10.1007/s42114-024-00879-6&api_key={os.getenv('SPRINGER_TDM_API_KEY')}"

try:
    print(url)
    # Send GET request with headers
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the response content to a file
        with open("springer_test.xml", "w", encoding="utf-8") as file:
            file.write(response.text)
    else:
        print(f"Request failed with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
