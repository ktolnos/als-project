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

# Get the column names by running the analysis on one file to see what it returns
sample_output = run_analysis_single(files[0], pr_script, tg_script)

columns = sample_output.columns

# Initialize the DataFrame with columns from sample_output
total_output = pd.DataFrame(columns=sample_output.columns)


for file in files:
    print(f"{count}.->{os.path.basename(file)}")

    out = run_analysis_single(file, pr_script, tg_script, get_acoustics)

    
    total_output = pd.concat([total_output, out])

    
    total_output.to_csv(os.path.join(os.path.dirname(file), "test_csv.csv"))
    count += 1

