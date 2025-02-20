import os
from dotenv import load_dotenv
import requests

load_dotenv()

scopus_api_key = os.getenv("SCOPUS_API_KEY")

# Specify the URL
url = "https://api.elsevier.com/content/article/doi/10.1016/j.mattod.2023.03.011"

# Specify the headers
headers = {"X-ELS-APIKey": scopus_api_key, "Accept": "application/xml"}
print(headers)  # Print the headers to the console

# Send the GET request and write the response to a file


try:
    # Send GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the response content to a file
        with open("elsevier_test.xml", "w", encoding="utf-8") as file:
            file.write(response.text)
    else:
        print(f"Request failed with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
