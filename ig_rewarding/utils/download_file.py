import requests
import os


def download_file_to_cache(url, cache_dir, cache_filename):
    # Create the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define the full path to the cache file
    cache_file_path = os.path.join(cache_dir, cache_filename)

    # Check if the file is already in the cache
    if os.path.exists(cache_file_path):
        print(f"File already exists in cache: {cache_filename}")
        return cache_file_path

    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to the cache file
        with open(cache_file_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved to cache: {cache_filename}")
        return cache_file_path
    else:
        print(f"Failed to download the file: {url}")
        return None
