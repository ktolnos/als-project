import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)

    # Truncate the frequency at 5 kHz and calculate Mel spectogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr, fmax=5000)

    # Convert spectogram to decibel scale
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close()


def create_augmented_spectogram(audio_file, image_file, augmented_image_file):

    # Open the saved image
    original_image = Image.open(image_file)

    # Flip the image horizontally
    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Save the flipped image
    flipped_image.save(augmented_image_file)


df_voiced = pd.read_csv("/Users/user/Desktop/github_repos/als-project/final_metadata_acoustic_features.csv")

updated_column_names = df_voiced.columns.to_list() + ["spectogram_file_path", "spectogram_type"]

relative_path = "/Users/user/Desktop/github_repos/als-project"
updated_rows = []

orig_spectogram = "spectogram"
aug_spectogram = "augmented_spectogram"

n = 0 
for idx, row in df_voiced.iterrows():
    file_path = row['file_path']
    full_path = os.path.join(relative_path, file_path )
   
    one_dir_up = os.path.dirname(full_path)

    orig_spect = os.path.join(os.path.join(one_dir_up, orig_spectogram),os.path.basename(full_path))
    orig_spect = orig_spect.replace(".wav", ".png")

    if not os.path.exists(os.path.join(one_dir_up, orig_spectogram)):
        os.makedirs(os.path.join(one_dir_up, orig_spectogram))

    aug_spect = os.path.join(os.path.join(one_dir_up, aug_spectogram), os.path.basename(full_path))
    aug_spect = aug_spect.replace(".wav",  ".png")


    if not os.path.exists(os.path.join(one_dir_up, aug_spectogram)):
        os.makedirs(os.path.join(one_dir_up, aug_spectogram))

    new_row_orig = row.to_list() + [orig_spect, "original"]
    new_row_aug = row.to_list() + [aug_spect, "augmented"]
    updated_rows.append(new_row_orig)
    updated_rows.append(new_row_aug)

    if not os.path.isfile(orig_spect):
        create_spectrogram(full_path, orig_spect)
    else:
        print("Original spectogram exists.")

    # Crop the original spectogram
     # Load image
    image = Image.open(orig_spect)

    # Crop specs
    left_crop = 20
    right_crop = 20
    top_crop = 45

    # Original dimensions
    width, height = image.size

    # New dimensions
    left = left_crop
    top = top_crop
    right = width - right_crop
    bottom = height

    # Crop image
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(orig_spect)

    if not os.path.isfile(aug_spect):
        create_augmented_spectogram(full_path, orig_spect, aug_spect)
    else:
        print("Augmented spectogram already exists.")
    
    print(f"{n}. {full_path}")
    n += 1


df_voiced_orig_aug_spects = pd.DataFrame(updated_rows, columns=updated_column_names)

df_voiced_orig_aug_spects.to_csv(os.path.join(relative_path,"spectogram_acoustic_features.csv"))

   

