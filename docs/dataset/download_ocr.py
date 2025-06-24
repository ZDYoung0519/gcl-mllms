import os
import json
import urllib.request as ureq
import concurrent.futures
import threading

# Set the file paths for your Google Drive
dataset_path = '/storage/public_datasets/OCRVQA/dataset.json'
images_path = '/storage/public_datasets/OCRVQA/images'
download = 1  # Set to 0 if images are already downloaded

# Load dataset json file
with open(dataset_path, 'r') as fp:
    data = json.load(fp)

# Initialize a counter and a lock for thread-safe counting
downloaded_count = 0
count_lock = threading.Lock()


# Function to download an image
def download_image(k):
    global downloaded_count
    imageURL = data[k]['imageURL']
    ext = os.path.splitext(imageURL)[1]
    outputFile = os.path.join(images_path, f'{k}{ext}')

    # Only download the image if it doesn't exist
    if not os.path.exists(outputFile):
        try:
            ureq.urlretrieve(imageURL, outputFile)

            with count_lock:
                downloaded_count += 1
                if downloaded_count % 100 == 0:
                    print(f'{downloaded_count} images downloaded.')
        except urllib.error.URLError as e:
            print(f'Error downloading {outputFile}: {e}')


# Download images using multiple threads
if download == 1:
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Create a thread pool and download the images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_image, data.keys())
