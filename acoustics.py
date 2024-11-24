# %% Load packages

import parselmouth
import os
from parselmouth.praat import call
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy import linalg
from typing import List
from python_speech_features import delta
from itertools import product
from scipy.stats import describe

from extraction import (
    compute_complexity_array
)


window_size = 0.025
time_step = 0.01

# %% Acoustics

# Helpers for dataframe construction in the main function, acstc_anlys
def get_f0(pitch: parselmouth.Pitch) -> pd.DataFrame:
    '''
    Extracts F0 (fundamental frequency) statistics from a Parselmouth Pitch object, returns a pd.DataFrame.

    Parameters
    ----------
    pitch : parselmouth.Pitch
        Pitch object extracted from a sound.

    Returns
    -------
    pd.DataFrame
        Contains F0 statistics.

    '''
    
    arr = pitch.selected_array["frequency"]
    return pd.DataFrame(
        {
            "f0_mean": call(pitch, "Get mean", 0, 0, "Hertz"),
            "f0_median": np.median(arr[arr != 0]), 
            "f0_sd": call(pitch, "Get standard deviation", 0, 0, "Hertz"),
            "f0_cv": call(pitch, "Get standard deviation", 0, 0, "Hertz")
            / call(pitch, "Get mean", 0, 0, "Hertz"),
            "f0_slope": pitch.get_mean_absolute_slope(),
            "f0_p5": call(pitch, "Get quantile", 0, 0, 0.05, "Hertz"),
            "f0_p95": call(pitch, "Get quantile", 0, 0, 0.95, "Hertz"),
            "f0_p5-95": float(call(pitch, "Get quantile", 0, 0, 0.95, "Hertz"))
            - float(call(pitch, "Get quantile", 0, 0, 0.05, "Hertz")),
        },
        index=[0],
    )

def get_formants(
    voiced: parselmouth.Sound, pitch: parselmouth.Pitch, formant_range: int = 5
) -> tuple:
    '''
    Extract formants 1 to <formant_range>. Returns a tuple of a panads dataframe 
    of formant measures and a list containing the values of the formants.

    Parameters
    ----------
    voiced : parselmouth.Sound
        A Parselmouth Sound object. Must be voiced-only, which can be extracted beforehand
        using get_voiced.
    pitch : parselmouth.Pitch
        A Parselmouth Pitch object.
    formant_range : int, optional
        Number of formants to extract. The default is 5.

    Returns
    -------
    tuple
        (pd.DataFrame containing values, list of raw formant value arrays).

    '''
    
    formants = voiced.to_formant_burg(
        window_length=window_size, time_step=time_step
    )  # window_length = 0.025 with a step of 10 ms

    # Use a loop to get the formant values and accumulate the relevant measures into the dict for the dataframe.
    formant_values = []
    formant_dict = {}
    for i in range(1, formant_range + 1):
        formant_value = np.asarray(
            [formants.get_value_at_time(formant_number=i, time=t) for t in pitch.xs()]
        )

        formant_values.append(formant_value)

        formant_dict[f"f{i}_mean"] = np.nanmean(formant_value)
        formant_dict[f"f{i}_median"] = np.nanmedian(formant_value)  #######
        formant_dict[f"f{i}_std"] = np.nanstd(formant_value)
        formant_dict[f"f{i}_prc_5"] = np.nanquantile(formant_value, q=0.05)
        formant_dict[f"f{i}_prc_95"] = np.nanquantile(formant_value, q=0.95)
        formant_dict[f"f{i}_prc_5_95"] = np.nanquantile(
            formant_value, q=0.95
        ) - np.nanquantile(formant_value, q=0.05)

    result = pd.DataFrame(formant_dict, index=[0])
    return result, formant_values


# For debugging: arrays = [pitch_values, formant1_values, formant2_values]
def get_complexity(arrays: list, plot: bool = False) -> int:
    """
    Calculate complexity of interaction between two input signals using PCA to
    estimate the number of eigenvalues in the complexity spectra that account
    for 95% of the variance in the spectrum. Builds off of the logic of Talkar
    et al. (2021) that described complexity as a vector of eigenvalues; here,
    we condense this down to a single value to enable easier incorporation into
    downstream dataframes.

    Parameters
    ----------
    arrays: list
        A list of numpy arrays, must all be the same length.
    plot: boolean, optional
        Do we want to plot a heatmap of the cross-correlations? The default is False.

    Returns
    -------
    complexity_idx: int
        An integer describing the complexity of the interaction between the two inputs.

    """
    # Meta-features: Talkar et al., 2021 (and many others) coordination features
    win = 15

    for ix_a, a in enumerate(arrays):
        if np.sum(np.isnan(a)) > 0:
            a1 = pd.Series(a).interpolate("cubic")
            arrays[ix_a] = a1.values

    arr = np.zeros((win * len(arrays), win * len(arrays)), np.float64)
    for ixm1, m1 in enumerate(arrays):
        for ixm2, m2 in enumerate(arrays):
            # Mathematically, this is equivalent to constructing two time-delay
            # embedding matrices (1 for each signal) and then doing a matrix-
            # wise Pearson correlation
            arr_0 = compute_complexity_array(win, m1, m2)
            arr[
                (ixm1 * win) : ((ixm1 + 1) * win), (ixm2 * win) : ((ixm2 + 1) * win)
            ] = arr_0

    # The logic here is that higher complexity = healthy (Talkar etc papers)
    # higher complexity = more components to explain x% of variance
    # This is, then, a quick and easy univariate means by which to extract the
    # complexity of the correlation structure
    pca = PCA()
    pca.fit_transform(arr)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    complexity_idx = np.where(cum_var > 0.95)[0][0]
    eig_vals = linalg.eig(arr)[0]

    if plot == True:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(arr)
        ax[0].set_xlabel("Lagged correlations")
        ax[0].set_ylabel("Lagged correlations")

        ax[1].plot(eig_vals, c="k", marker="o")
        ax[1].set_xlabel("Eigenvalues of correlation matrix")
        ax[1].scatter(
            complexity_idx, eig_vals[complexity_idx], s=100, c="r", ec="k", zorder=3
        )
        ax[1].axvline(
            complexity_idx,
            ymax=eig_vals[complexity_idx] / eig_vals[0] * 1.1,
            linestyle=":",
            color="r",
            linewidth=1,
            zorder=2,
        )
        ax[1].text(
            complexity_idx,
            1.25,
            f"Complexity index: {complexity_idx}\n95% var explained",
            ha="left",
            va="bottom",
        )
        ax[1].text(complexity_idx, 0, "→ more complex", ha="left", va="center")

    return complexity_idx

def get_complexity_measures(arrays: dict, array_pairs: List[tuple]) -> pd.DataFrame:
    """
    Returns a pandas dataframe of coordination complexity measures (Talkar et al., 2021).

    Parameters
    ----------
    arrays (dict[np.ndarray]): A dictionary of arrays from which complexity measures are to be extracted.
    array_pairs (List[tuple]): A list of tuples representing the pairs of arrays from <arrays> to be used to calculate complexity.

    Returns
    -------
    (pd.DataFrame): A Pandas dataframe that contains the complexity measures obtained using the pairs of arrays specified.

    """
    complexity_measures = {}

    for pair in array_pairs:
        first = pair[0]
        second = pair[1]
        try:
            complexity_measures[f"{first}_{second}_comp"] = get_complexity(
                [arrays[first], arrays[second]]
            )
        except KeyError:
            print(f"{first} or {second} are not valid arrays!")

    return pd.DataFrame(complexity_measures, index=[0])


def get_formant_accels(
    formant_slope_values: list, times: np.ndarray, num_formants: int = 3
) -> pd.DataFrame:
    '''
    Extracts the number of formant accelerations as specified by <num_formants>. 
    Accelerations are extracted using the slope values specified by <formant_slope_values> 
    and time points specified by <times>. Returns a pandas dataframe of formant 
    acceleration measures.
    
    Parameters
    ----------
    formant_slope_values : list
        List of formant slope (i.e., first derivative) arrays extracted previously 
        using get_formant_slopes.
    times : np.ndarray
        Array of times at which to extract the formant values.
    num_formants : int, optional
        The number of formants to extract accelerations for. The default is 3.

    Returns
    -------
    pd.DataFrame
        Dataframe containing all formant acceleration values.

    '''
    formant_accels_dict = {}

    # Loop <num_formants> times to compute formant accels
    for i in range(num_formants):
        formant_accel = np.gradient(formant_slope_values[i], times * 20)

        formant_num = i + 1

        formant_accels_dict[f"f{formant_num}_dd_dx_median"] = np.nanmedian(
            formant_accel
        )
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_5"] = np.nanquantile(
            formant_accel, q=0.05
        )
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_95"] = np.nanquantile(
            formant_accel, q=0.95
        )
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_5_95"] = np.nanquantile(
            formant_accel, q=0.95
        ) - np.nanquantile(formant_accel, q=0.05)

    return pd.DataFrame(formant_accels_dict, index=[0])

def get_formant_slopes(formant_values: list, times: np.ndarray, num_formants: int = 3) -> tuple:
    '''
    Extracts the number of formant slopes (i.e., first derivatives) as specified by <num_formants>. 
    Slopes are extracted using the slope values specified by <formant_values> 
    and time points specified by <times>. Returns a pandas dataframe of formant 
    slope measures.
    
    Parameters
    ----------
    formant_slope_values : list
        List of formant slope (i.e., first derivative) arrays extracted previously 
        using get_formant_slopes.
    times : np.ndarray
        Array of times at which to extract the formant values.
    num_formants : int, optional
        The number of formants to extract slopes for. The default is 3.

    Returns
    -------
    pd.DataFrame
        Dataframe containing all formant slope values.

    '''
    
    formant_slopes = []
    formant_slopes_dict = {}

    # Loop <num_formants> times to compute formant slopes
    for i in range(num_formants):

        formant_slope = np.gradient(formant_values[i], times * 20)

        formant_slopes.append(formant_slope)

        formant_num = i + 1
        formant_slopes_dict[f"f{formant_num}_d_dx_median"] = np.nanmedian(formant_slope)
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_5"] = np.nanquantile(
            formant_slope, q=0.05
        )
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_95"] = np.nanquantile(
            formant_slope, q=0.95
        )
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_5_95"] = np.nanquantile(
            formant_slope, q=0.95
        ) - np.nanquantile(formant_slope, q=0.05)

    result = pd.DataFrame(formant_slopes_dict, index = [0])

    return result, formant_slopes

def get_jitter(pointProcess: parselmouth.Data) -> pd.DataFrame:
    '''
    Extracts various jitter features from a Parselmouth PointProcess object.

    Parameters
    ----------
    pointProcess : parselmouth.Data
        PointProcess that was previously generated from a Parselmouth Sound object.

    Returns
    -------
    pd.DataFrame
        Contains jitter statistics for provided sample.

    '''
    
    return pd.DataFrame(
        {
            "rapJitter": (
                call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            )
            * 100,
            "localJitter": (
                call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            )
            * 100,
            "localabsoluteJitter": (
                call(
                    pointProcess,
                    "Get jitter (local, absolute)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                )
            )
            * 100,
            "ppq5Jitter": (
                call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            )
            * 100,
            "ddpJitter": (
                call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            )
            * 100,
        },
        index=[0],
    )


def get_MFCCs(sound: parselmouth.Sound) -> pd.DataFrame:
    """
    Returns a pandas dataframe with MFCC measures.

    Parameters
    ----------
    sound: (parselmouth.Sound)
        A parselmouth.Sound object created from raw data. e.g., sound = parselmouth.Sound(path/to/file).

    Returns
    -------
    (pd.DataFrame) 
    A Pandas dataframe that contains the mel-frequency cepstral coefficients (MFCCs). Note that as 
    per Parselmouth/Praat, the first feature is energy, and the remaining n features are true MFCCs. 
    Thus, the default output is 1 energy feature + 13 MFCC feature sets.

    """

    n_mfcc = (
        1 + 13
    )
    mfcc_object = sound.to_mfcc(number_of_coefficients=n_mfcc - 1)
    mfcc0 = mfcc_object.to_array().T
    mfccs = np.c_[mfcc0, delta(mfcc0, 2)]
    tsc_mfcc = np.sum(np.gradient(mfcc0[:, 1:], axis=0) ** 2, axis=1)

    tsc_mfcc_df = pd.DataFrame(
        {
            "tsc_mfcc_mean": tsc_mfcc.mean(),
            "tsc_mfcc_cv": tsc_mfcc.std() / tsc_mfcc.mean(),
        },
        index=[0],
    )

    mfcc_df = pd.DataFrame(
        data=mfccs,
        columns=["mfcc_" + str(k) for k in range(n_mfcc)]
        + ["mfcc_slope_" + str(k) for k in range(n_mfcc)],
    )
    _, _, mu, sig, _, _ = describe(mfccs, axis=0)
    mfcc_stats = np.r_[mu, sig]
    mfcc_summary_df = pd.DataFrame(
        data=mfcc_stats.T,
        index=[
            "_".join([x, y]) for x, y in list(product(["mean", "var"], mfcc_df.columns))
        ],
    ).T

    return pd.concat([tsc_mfcc_df, mfcc_summary_df], axis=1)


def get_harmonicity(voiced: parselmouth.Sound) -> pd.DataFrame:
    """
    Returns a pandas dataframe containing harmonicity mean and SD measures.

    Parameters
    ----------
    voiced: parselmouth.Sound
        A Parselmouth Sound object representing the audio data you wish to analyze.

    Returns
    -------
    pd.DataFrame
        A Pandas Dataframe containing HNR (referred to as harmonicity in Praat/Parselmouth) mean and SD values.

    """
    harmonicity = call(
        voiced, "To Harmonicity (cc)", 0.01, 75, 0.1, 1
    )  # https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__ac____.html

    return pd.DataFrame(
        {
            "hnr_mean": call(harmonicity, "Get mean", 0, 0),
            "hnr_sd": call(harmonicity, "Get standard deviation", 0, 0),
        },
        index=[0],
    )


def get_intensity(voiced: parselmouth.Sound) -> pd.DataFrame:
    """
    Returns a pandas dataframe with intensity mean, SD, and CV measures.

    Parameters
    ----------
    voiced: parselmouth.Sound)
        A parselmouth.Sound object created from raw data. e.g., sound = parselmouth.Sound(path/to/file).

    Returns
    -------
    pd.DataFrame: 
        A Pandas dataframe summarizing the intensity of the input sample.

    """

    intensity = call(voiced, "To Intensity...", 100, 0.0)

    return pd.DataFrame(
        {
            "intensity_mean_dB": call(intensity, "Get mean...", 0.0, 0.0, "dB"),
            "intensity_sd_dB": call(
                intensity, "Get standard deviation...", 0.0, 0.0
            ),  # dB is the default
            "intensity_cv_dB": call(intensity, "Get standard deviation...", 0.0, 0.0)
            / call(intensity, "Get mean...", 0.0, 0.0, "dB"),
        },
        index=[0],
    )


def interp_formant(formant_values: np.ndarray) -> np.ndarray:
    """
    A function to interpolate missing values in formant arrays using linear interpolation from Pandas.

    Parameters
    ----------
    formant_values: np.ndarray
        An array of formant values to be interpolated.

    Returns
    -------
    formant_values: np.ndarray
        Modified version of the input array that has been interpolated (missing value handling).

    """
    if any(np.isnan(formant_values)):
        print(f"Interpolated {sum(np.isnan(formant_values))} value(s)")
        formant_values = (
            pd.Series(formant_values)
            .interpolate(method="linear", limit_direction="both")
            .values
        )
    return formant_values


def nearest(short_arr: np.ndarray, long_arr: np.ndarray, tol = 0.001) -> np.ndarray:
    """
    A function to retrieve temporal alignments (of timestamps) between a shorter array (with coarser sampling)
    and a longer array (with more granular sampling). The objective is to find the entries in the
    shorter array that exist in the longer array, within a tolerance (of 0.001 sec as default).

    Parameters
    ----------
    short_arr: np.ndarray
        Array of timestamps representing the more coarsely-sampled modality to be aligned.
    long_arr: np.ndarray
        Array of timestamps representing the more granularly-sampled modality to be aligned.
    tol: float
        A float describing the maximum amount of difference between timestamps in both arrays for them
        to be considered "the same". e.g., 1.001 sec could be considered the same as 1.000 sec
        if tol>0.001, but not if tol<=0.001.
    Returns
    -------
    np.ndarray
        An array of the values in the shorter array that matched the entries in the longer array.

    """
    match_vals = []
    for s in short_arr:
        new_val = np.argmin(np.abs(s - long_arr))
        if np.min(np.abs(s - long_arr)) > tol:
            print("INTENSITY: Detected delta >1ms")
        match_vals.append(new_val)
    return np.asarray(match_vals).ravel()


def get_shimmer(
    voiced: parselmouth.Sound, pointProcess: parselmouth.Data
) -> pd.DataFrame:
    '''
    Extracts various shimmer features from a Parselmouth PointProcess object and onlyVoiced file
    read in as a Parselmouth Sound object.

    Parameters
    ----------
    pointProcess : parselmouth.Data
        PointProcess that was previously generated from a Parselmouth Sound object.

    Returns
    -------
    pd.DataFrame
        Contains shimmer statistics for provided sample.

    '''
    return pd.DataFrame(
        {
            "localShimmer": (
                call(
                    [voiced, pointProcess],
                    "Get shimmer (local)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            * 100,
            "localdbShimmer": call(
                [voiced, pointProcess],
                "Get shimmer (local_dB)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            ),
            "apq3Shimmer": (
                call(
                    [voiced, pointProcess],
                    "Get shimmer (apq3)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            * 100,
            "aqpq5Shimmer": (
                call(
                    [voiced, pointProcess],
                    "Get shimmer (apq5)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            * 100,
            "apq11Shimmer": (
                call(
                    [voiced, pointProcess],
                    "Get shimmer (apq11)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            * 100,
            "ddaShimmer": (
                call(
                    [voiced, pointProcess],
                    "Get shimmer (dda)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            * 100,
        },
        index=[0],
    )

def acstc_anlys(
    file: str,
    data_dir: str,
    f0_min: int = 75,
    f0_max: int = 600,
) -> pd.DataFrame:
    """
    Function for performing a comprehensive acousti analysis on an inputted sample.

    Parameters
    ----------
    file: str
        The absolute path to the audio file we want to analyze.
    data_dir: str
        The path to the folder in which the audio file is found.
    f0_min: int, optional
        Minimum frequency to consider for a given sample's fundamental frequency.
    f0_max: int, optional
        Maximum frequency to consider for a given sample's fundamental frequency.

    Returns
    -------
    pd.DataFrame
        Contains all analysis results for the given sample.

    """

    # Get relevant variables
    f_old = os.path.basename(file)
    fnew = f_old.replace(".wav", "_OnlyVoiced.wav")
    sound = parselmouth.Sound(f"{data_dir}/{f_old}")
    voiced = parselmouth.Sound(f"{data_dir}/voiced/{fnew}")

    # Extract F0 and formants - ignore calling it "pitch"
    pitch = call(
        voiced,
        "To Pitch (cc)",
        0.02,
        f0_min,
        15,
        "no",
        0.03,
        0.45,
        0.01,
        0.35,
        0.14,
        f0_max,
    )
    pitch_values = pitch.selected_array["frequency"]
    pitch_times = np.asarray(pitch.xs())

    # Used for jitter and shimmer
    pointProcess = call([voiced, pitch], "To PointProcess (cc)")

    jitter_df = get_jitter(pointProcess)
    shimmer_df = get_shimmer(voiced, pointProcess)
    harmonicity_df = get_harmonicity(voiced)
    f0_df = get_f0(pitch)
    formants_df, formant_values = get_formants(voiced, pitch)
    formant_slopes_df, formant_slope_values = get_formant_slopes(
        formant_values, pitch_times
    )
    formant_accels_df = get_formant_accels(formant_slope_values, pitch_times)
    intensity_df = get_intensity(voiced)
    mfccs_df = get_MFCCs(sound)
   

    voiced_norm = voiced.copy()
    call(voiced_norm, "Scale intensity...", 70.0)
    intensity_norm = call(
        voiced_norm, "To Intensity...", 100, 0.001
    )  # Set this timestep so that we're guaranteed to get values that align to the formants' sampling times
    match_vals = nearest(short_arr=pitch_times, long_arr=intensity_norm.xs())
    intensities_final = intensity_norm.values[0][match_vals]

    # The arrays to be compared to get complexity measures
    comp_arrays = {
        "F0": pitch_values,
        "F1": interp_formant(formant_values[0]),
        "F2": interp_formant(formant_values[1]),
        "F3": interp_formant(formant_values[2]),
        "IN": intensities_final,
    }

    comp_array_pairs = [
        ("F0", "F1"),
        ("F0", "F2"),
        ("F0", "F3"),
        ("F1", "F2"),
        ("F1", "F3"),
        ("F2", "F3"),
        ("F0", "IN"),
    ]

    # Call helper to get the dataframe for complexities
    complexities_df = get_complexity_measures(comp_arrays, comp_array_pairs)

    # Put dataframes together, return
    return pd.concat(
        [
            jitter_df,
            shimmer_df,
            harmonicity_df,
            f0_df,
            formants_df,
            formant_slopes_df,
            formant_accels_df,
            intensity_df,
            mfccs_df,
            complexities_df,
        ],
        axis=1,
    )
