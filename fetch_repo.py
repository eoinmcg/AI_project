import argparse
import os
import requests
import csv
from urllib.parse import urljoin

from llama_index.core import Document

FILE="repo_files.csv"

def get_github_repo_files(repo_url):
    """
    Fetches all JavaScript files in the 'src' directory of a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Returns:
        list: A list of dictionaries, each containing file information (filename, source, url).
    """
    try:
        repo_name = repo_url.split("github.com/")[1].rstrip("/")
        api_url = f"https://api.github.com/repos/{repo_name}/contents/src"
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        files = response.json()
        js_files = []

        for file in files:
            if file["type"] == "file" and file["name"].endswith(".js"):
                file_info = {
                    "filename": file["name"],
                    "url": file["download_url"],
                }
                file_content_response = requests.get(file["download_url"])
                file_content_response.raise_for_status()
                file_info["source"] = file_content_response.text
                js_files.append(file_info)

        return js_files

    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository files: {e}")
        return []
    except IndexError:
        print("Invalid GitHub repository URL format.")
        return []
    except KeyError:
        print("src directory not found in the repository")
        return []
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return []

def write_to_csv(data, output_file=FILE):
    """
    Writes file information to a CSV file.

    Args:
        data (list): A list of dictionaries, each containing file information.
        output_file (str): The name of the output CSV file.
    """
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "source", "url"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for file_info in data:
                writer.writerow(file_info)
        print(f"File information written to {output_file}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def create_csv():
    """
    Main function to parse arguments and process the repository.
    """
    parser = argparse.ArgumentParser(description="Fetch JavaScript files from a GitHub repository's src directory.")
    parser.add_argument("repo_url", help="The GitHub repository URL.")

    repo_url = "https://github.com/KilledByAPixel/LittleJS"

    files = get_github_repo_files(repo_url)
    if files:
        write_to_csv(files)


print('READY?')

rows = []
# Load the file as a JSON
with open(FILE, mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
        if idx == 0: continue; # Skip header row
        rows.append(row)

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row[1], metadata={"title": row[0], "url": row[2]}) for row in rows]
print(len(documents))
print(documents[0].metadata)
print(documents[0])


