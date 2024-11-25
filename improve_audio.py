#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import pandas as pd
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

# GitHub configuration
GITHUB_REPO_URL = "https://github.com/ktolnos/als-project.git"  # Your GitHub repository URL
GITHUB_TOKEN = "Your_Token"  # Replace with your token
LOCAL_REPO_DIR = "/Users/fletcher/Desktop/als-project"  # Local clone of the repository

# Input CSV file
csv_file = "final_metadata_acoustic_features.csv" # input the csv ciantaining file paths of original audios

# Parameters
target_sr = 16000  # Target sample rate for resampling
pre_emphasis_coeff = 0.97
low_cutoff = 300.0  # Low-pass cutoff frequency in Hz
high_cutoff_ratio = 0.99  # High cutoff as a ratio of Nyquist frequency


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)  # Ensure high cutoff is < Nyquist
    if low <= 0 or high <= low or high >= 1:
        raise ValueError(
            f"Filter critical frequencies must be 0 < low < high < Nyquist. "
            f"Given lowcut={lowcut}, highcut={highcut}, nyquist={nyquist}."
        )
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, fs, highcut_ratio=0.99, order=5):
    nyquist = 0.5 * fs
    highcut = highcut_ratio * nyquist  # Scale high cutoff by ratio
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def pre_emphasize(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def clone_or_pull_repo(repo_url, local_dir, token):
    """Clone the GitHub repository if it doesn't exist, or pull changes if it does."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        clone_url = repo_url.replace("https://", f"https://{token}@")
        subprocess.run(["git", "clone", clone_url, local_dir], check=True)
        print(f"Cloned repository to {local_dir}")
    else:
        subprocess.run(["git", "-C", local_dir, "pull"], check=True)
        print(f"Pulled latest changes into {local_dir}")


def process_and_save_files(csv_file, repo_dir):
    # Read the CSV file
    metadata = pd.read_csv(csv_file)

    # Assuming the file paths are in the second column
    file_paths = metadata.iloc[:, 2].dropna()

    # Process files
    for file_path in file_paths:
        try:
            # Define local file path
            local_file_path = os.path.join(repo_dir, file_path.strip())
            if not os.path.exists(local_file_path):
                print(f"File not found locally: {local_file_path}")
                continue

            # Load audio
            y, sr = librosa.load(local_file_path, sr=None)

            # Resample
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Apply bandpass filter
            y = bandpass_filter(y, lowcut=low_cutoff, fs=sr, highcut_ratio=high_cutoff_ratio, order=5)

            # Apply pre-emphasis
            y = pre_emphasize(y, coeff=pre_emphasis_coeff)

            # Define output file path to avoid overwriting the original file
            output_file_path = os.path.join(repo_dir, "filtered", file_path.strip())
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Save improved audio
            sf.write(output_file_path, y, sr)
            print(f"Processed and saved: {output_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def commit_and_push_changes(local_dir, commit_message="Flitered audio files"):
    """Commit and push changes to the GitHub repository."""
    try:
        subprocess.run(["git", "-C", local_dir, "add", "filtered/"], check=True)
        subprocess.run(["git", "-C", local_dir, "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "-C", local_dir, "push"], check=True)
        print("Changes pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error during git operation: {e}")


# Main execution
if __name__ == "__main__":
    # Step 1: Clone or pull the repository
    clone_or_pull_repo(GITHUB_REPO_URL, LOCAL_REPO_DIR, GITHUB_TOKEN)

    # Step 2: Process the files and save in the local repository
    process_and_save_files(csv_file, LOCAL_REPO_DIR)

    # Step 3: Commit and push the changes back to the GitHub repository
    commit_and_push_changes(LOCAL_REPO_DIR)


# In[ ]:




