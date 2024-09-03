import os
import subprocess
import zipfile

DATASETS_PATH = "datasets/kitti"

def download_dataset(url, path):
    try:
        print(f"Downloading dataset from {url}" to {path})
        subprocess.run(["wget", "-O", path, url], check=True)
        print(f"Dataset downloaded to {path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset from {url}: {e}")
        os.remove(path)
        raise e

def extract_dataset(path):
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            print(f"Extracting dataset from {path} to {path}")
            zip_ref.extractall(path_path)
            print(f"Dataset extracted to {path}")
    except Exception as e:
        print(f"Failed to extract dataset from {path}: {e}")
        raise e

def download_and_extract_dataset(dataset_type, path):
    os.makedirs(path, exist_ok=True)
    
    if dataset_type == "kitti":
        urls = [
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip",
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
        ]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    for url in urls:
        download_dataset(url, path)
        extract_dataset(path)

if __name__ == "__main__":
    download_and_extract_dataset("kitti", DATASETS_PATH)
