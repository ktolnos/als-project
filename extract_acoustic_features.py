import numpy as np
import pandas as pd
import glob
import os
from acoustics import acstc_anlys
from utilities import get_voiced
import time

def run_analysis_single(f, pr_script, tg_script, get_acoustics=True):
    '''
    A function to analyze 

    Parameters
    ----------
    f : str
        Filepath to -converted.wav audio file (absolute path). MUST be the audio file
    pr_script : str
        Praat script for extracting voiced segments from audio data (absolute path).
    tg_script : str
        Praat script for textgrid parsing for timing analysis (absolute path).
        Praat script for analyzing pauses from audio data (absolute path).
    get_acoustics : bool
        Extract acoustic features?

    Returns
    -------
    output : pd.DataFrame
        Single-row dataframe containing all extracted features.

    '''
    
    # Verify inputs
    assert (type(pr_script)==str) & (len(pr_script)>3), "Please specify location of voice extraction Praat script"

    file = f.replace("\\", "/")
    data_dir = os.path.dirname(file).replace("\\", "/")
    
    fnew_test = os.path.dirname(file) + "/" + "voiced/" + os.path.basename(file).replace(".wav", "_OnlyVoiced.wav")
    if get_acoustics==True:
        if os.path.isfile(fnew_test):
            print("Voiced file already extracted")
            pass
        else:
            get_voiced(file = file, 
                                        data_dir = data_dir,
                                        praat_script = pr_script,
                                        textgrid_script = tg_script
                                        )
        # get acoustic features
        output = acstc_anlys(file = file,
                                    data_dir = data_dir,
                                    f0_min = 75,
                                    f0_max = 600
                                    )
    
    output.index = [os.path.basename(file).split('.')[0]]
    
    return output    

cfg = pd.read_csv('config.config', header=None, index_col=0)
os.chdir(cfg.loc["run_home"].values[0])
folder = cfg.loc["run_home"].values[0]
tg_script = cfg.loc["textgrid_script"].values[0]
pr_script = cfg.loc["praat_script"].values[0]

files = glob.glob(folder+"*.wav", recursive=True)

count=1

# Upload the final consolidated  dataframe
metadata_path = "/Users/user/Desktop/github_repos/als-project/filtered/data/filtered_final_consolidated_dataset_Fletcher.csv"
df = pd.read_csv(metadata_path)
relative_path = os.path.dirname(os.path.dirname(os.path.dirname(metadata_path)))


# Get the column names by running the analysis on one file to see what it returns
first_path = df.loc[944,"filtered_file_path"]
first_path =  os.path.join(relative_path, first_path)
sample_output = run_analysis_single(first_path, pr_script, tg_script)

new_columns = df.columns.to_list() + sample_output.columns.to_list()

# Initialize the DataFrame with columns from sample_output

combined_rows = []
error = []

df_948 = df.iloc[945:,:]


for index, row in df_948.iterrows():

    file_path = os.path.join(relative_path, row['filtered_file_path'])
    print(f"{count}  ->  {os.path.basename(file_path)}")

    try:
        out = run_analysis_single(file_path, pr_script, tg_script)
        combined_row = list(row) +  list(out.iloc[0,:])
        combined_rows.append(combined_row)

        print(combined_row)

        total_output = pd.DataFrame(combined_rows, columns=new_columns)
        total_output.to_csv(os.path.join(relative_path, "filtered_metadata_acoustic_clinical.csv"))   

        count += 1
    except:
        error.append(file_path)

error_df = pd.DataFrame(error, columns="error")
error_df.to_csv(os.path.join(relative_path, "error_files.csv"))

    



