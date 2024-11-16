# %% Load packages

import numpy as np
from pathlib import Path
import parselmouth

import os
import pandas as pd

# %% General utilities functions


def get_voiced(file: str, data_dir: str , praat_script: str, textgrid_script: str ) -> tuple:
    '''
    Extracts voiced audio from raw audio data. Uses a Praat script to achieve this.
    Creates a /voiced subfolder in the directory where the audio data live,
    if one doesn't already exist (will not remove/modify existing files)

    Parameters
    ----------
    file (str): Name of file to be analysed.
    data_dir (str, optional): Directory where the data to be analysed reside. 
    praat_script (str, optional): Directory where the extract_voiced_segments.praat Praat script resides. 
    textgrid_script (str, optional): Praat script for extracting textgrids from Praat and converting them to Python-readable inputs.

    Returns
    -------
    (tuple): A tuple with four values
        sound (parselmouth.Sound): A parselmouth.Sound object corresponding to the raw input file.
        voiced (parselmouth.Sound): A parselmouth.Sound object corresponding to the voiced only version of the file.
        rate (int): Sampling frequency (Hz) of the input file.
        SNR (float): Value representing the signal-to-noise ratio (SNR) of the audio.
        

    '''

    Path(f'{data_dir}/voiced/').mkdir(parents=True, exist_ok=True)
    f_old = os.path.basename(file)

    try:
        textgrid_out = parselmouth.praat.run_file(textgrid_script,
                                                  f'{data_dir}/{f_old}', 50, 0.0, -25.0, 0.01, 0.01, "silent", "sounding",
                                                  return_variables=True) 
        start_time = textgrid_out[1]['start']
    except:
        start_time = 0.1
        # Temp to keep track of files that are especially noisy, which cause an error with textgrid files
        print("Interval num error")
        
    outp = parselmouth.praat.run_file(praat_script,
                                      "/".join(file.split("/")[:-1]),
                                      f_old,
                                      "0", 
                                      str(start_time), 
                                      return_variables=True)
    
    pA_noise = outp[1]['maxPAsilence']
    pA_signal = outp[1]['maxPA']
   
    SNR = 20*np.log10(pA_signal/pA_noise)
   
    # Make new voiced directory if it doesn't already exist
    fnew        = f_old.replace('.wav', '_OnlyVoiced.wav')
   
    sound       = parselmouth.Sound(f'{data_dir}/{f_old}')
    voiced      = parselmouth.Sound(f'{data_dir}/voiced/{fnew}')
    rate        = sound.sampling_frequency

    return sound, voiced, rate, SNR


    # file_name=kfile
    try:
        assert (".mp4" in file_name)
        csv_fname = file_name.replace('.mp4', '-tstamps.csv')
    except:
        assert (".webm" in file_name)
        csv_fname = file_name.replace('.webm', '-tstamps.csv')
    
    if os.path.isfile(csv_fname):
        print("already done")
    else:
        # break
        # file_name = 'C:/Users/leifs/Downloads/tibd55.webm'
        container = av.open(file_name)
        
        tstamps = list()
        # break
        for i, frame in enumerate(container.decode(video=0)):    
            if i%100==0:
                print(f'Tstamp {i}')
            tstamps.append(frame.time)
        # out=frame.to_image()
        tstamps = np.asarray(tstamps)
        
        time_arr = np.zeros((len(tstamps), 2))
        time_arr[:, 0] = np.arange(len(tstamps))
        time_arr[:, 1] = tstamps
        
        df = pd.DataFrame(time_arr)
        df.columns = ['frame', 'timestamp']
        df.to_csv(csv_fname, index = None)    