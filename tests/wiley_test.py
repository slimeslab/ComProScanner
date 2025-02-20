import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Specify the URL
url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1111/1467-923X.12168"

headers = {"Wiley-TDM-Client-Token": os.getenv("WILEY_API_KEY")}

try:
    # Send GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the response content to a file
        with open(
            "wiley_test.pdf", "wb"
        ) as file:  # Wiley API doesn't return XML, it returns PDF only
            file.write(response.content)
    else:
        print(f"Request failed with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
