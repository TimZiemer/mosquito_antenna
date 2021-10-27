from math import pi, cos, sin
from time import perf_counter
from typing import List  # For compatibility of type hinting with Python 3.8 or earlier

import librosa
import numpy as np
import scipy.signal

from visualizations import animate_antenna_movement, create_slider_for_time_series_and_db_spec, \
    create_multicolor_plot_for_paper, antiphase_prong_plot, magnitude_of_nonlinearities_plot
from utils import calculate_points, capsule_func, capsule_func_2, \
    create_mosquito_sound, calculate_db_spec_for_time_series, save_specific_frequencies, create_simple_sine_wave,\
    create_time_series_and_antenna_coordinates


def localization(degrees: List[int], sampling_rate: int, samples: int, f0_1: int, f0_2: int, secs: int) -> None:
    """
    Performs an experiment to determine whether or not the amplitude differences in the original signal
    transfer to the nonlinear product of the mosquito antennae. To get accurate results, a bin size of atleast 2 Hz
    should be chosen for FFT.
    :param degrees: list of ints, containing the degrees in which the sound sources are positioned,
    relative to the mosquito
    :param sampling_rate: int, sampling rate of the mosquito sounds
    :param samples: how many samples to look at
    :param f0_1: fundamental frequency of "our" mosquito
    :param f0_2: fundamental frequency of "external sound source" mosquito
    :param secs: duration of mosquito sounds in seconds
    """
    distance_between_antennas = 0.125  # mm  Source: Göpfert and Robert 2001
    radius = 15  # mm
    # Center of the circle is seen as the middle between the antennas

    # Calculate distance for each antenna
    x_a2, y_a2 = -(distance_between_antennas / 2), 0
    x_a1, y_a1 = distance_between_antennas / 2, 0  # A1 is closer to the sound source

    for deg in degrees:
        # Calculate sound source locations on the circle
        # Source: https://math.stackexchange.com/questions/260096/find-the-coordinates-of-a-point-on-a-circle
        x, y = radius * sin(np.deg2rad(deg)), radius * cos(np.deg2rad(deg))

        r_a1 = np.linalg.norm(np.array([[x, y]]) - np.array([[x_a1, y_a1]]), axis=1)
        r_a2 = np.linalg.norm(np.array([[x, y]]) - np.array([[x_a2, y_a2]]), axis=1)
        r_a1 = 1 / r_a1
        r_a2 = 1 / r_a2

        # Create signals
        sr, SIGNAL = create_mosquito_sound(sampling_rate, f0_1, secs, 1)  # Our mosquito
        _, SIGNAL1 = create_mosquito_sound(sampling_rate, f0_2, secs, r_a1)  # External sound source
        _, SIGNAL1_2 = create_mosquito_sound(sampling_rate, f0_2, secs, r_a2)  # External sound source

        SIGNAL_D_1 = SIGNAL.copy() + SIGNAL1
        SIGNAL_D_1 = SIGNAL_D_1 / 3  # Central normalization with constant factor
        SIGNAL_D_2 = SIGNAL.copy() + SIGNAL1_2
        SIGNAL_D_2 = SIGNAL_D_2 / 3

        # Time series etc. for first antenna
        time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
            SIGNAL_D_1[:samples], CAPSULE_COORDINATES)
        db_specs = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
        db_specs_orig = calculate_db_spec_for_time_series(SIGNAL_D_1[:samples], sr, zero_scale=True)
        save_specific_frequencies(db_specs, 1, deg, [72, 164, 236], 2)
        create_slider_for_time_series_and_db_spec(pointwise_time_series, SIGNAL_D_1[:samples], db_specs, db_specs_orig)

        # Time series etc. for second antenna
        time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
            SIGNAL_D_2[:samples], CAPSULE_COORDINATES)
        db_specs = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
        db_specs_orig = calculate_db_spec_for_time_series(SIGNAL_D_2[:samples], sr, zero_scale=True)
        save_specific_frequencies(db_specs, 2, deg, [72, 164, 236], 2)
        create_slider_for_time_series_and_db_spec(pointwise_time_series, SIGNAL_D_2[:samples], db_specs, db_specs_orig)


def inspect_real_single_mosquito(audio_file: str, overlay_file: str = None, cutoff: float = 1, sr: int = None) -> None:
    """
    Function designed to examine a single non-synchronizing mosquito audio.
    It is possible to create an overlay to simulate something similar to synchronization.
    :param audio_file: path to mosquito-audio file
    :param overlay_file: path to overlay-audio file
    :param cutoff: how many seconds to inspect
    :param sr: desired sampling rate for the audio, leave at None to keep original sr
    """
    data, samplerate = librosa.load(audio_file, sr=sr)
    b, a = scipy.signal.butter(N=2, Wn=[300, 2048], btype='bandpass', fs=samplerate)
    data = scipy.signal.lfilter(b, a, data)
    if overlay_file:
        data1, samplerate1 = librosa.load(overlay_file, sr=sr)
        b, a = scipy.signal.butter(N=2, Wn=[300, 2048], btype='bandpass', fs=samplerate)
        data1 = scipy.signal.lfilter(b, a, data1)
        if data.shape[0] != data1.shape[0]:
            min_length = min(data.shape[0], data1.shape[0])
            data1 = data1[:min_length]
            data = data[:min_length]
        data = data + data1
        data = data / abs(max(data))
    else:
        pass
    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(data[:int(samplerate*cutoff)], CAPSULE_COORDINATES)
    animate_antenna_movement(antenna_coords, CAPSULE_COORDINATES)
    db_specs =calculate_db_spec_for_time_series(pointwise_time_series, samplerate, zero_scale=False, antenna=True)
    db_specs_orig =calculate_db_spec_for_time_series(data[:int(samplerate*cutoff)], samplerate, zero_scale=False)
    create_slider_for_time_series_and_db_spec(pointwise_time_series, data[:int(samplerate*cutoff)], db_specs, db_specs_orig)


def inspect_synch_mosquito_audio(audio_file: str, cutoff: float = 1, scale: int = 1, sr: int = None) -> None:
    data, sr = librosa.load(audio_file, sr=sr)
    #b, a = scipy.signal.butter(N=2, Wn=[500, 2048], btype='bandpass', fs=sr)
    #data = scipy.signal.lfilter(b, a, data)
    data = data * scale  # Upscale the audio magnitude by some factor int(sr*cutoff)
    print(min(data), max(data))  # Could be useful information
    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(data[:int(sr*cutoff)], CAPSULE_COORDINATES)
    animate_antenna_movement(antenna_coords, CAPSULE_COORDINATES)
    db_specs =calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
    db_specs_orig =calculate_db_spec_for_time_series(data[:int(sr*cutoff)], sr, zero_scale=True)
    create_slider_for_time_series_and_db_spec(pointwise_time_series, data[:int(sr*cutoff)], db_specs, db_specs_orig)


### Following are experiments and plots for the antenna paper ###

def single_sine_wave_experiment():
    sr, signal = create_simple_sine_wave(22050, 1200, 1, 1)
    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
        signal[:2205], CAPSULE_COORDINATES)
    db_specs = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
    db_specs_orig = calculate_db_spec_for_time_series(signal[:2205], sr, zero_scale=True)
    create_slider_for_time_series_and_db_spec(pointwise_time_series, signal[:2205], db_specs, db_specs_orig)
    animate_antenna_movement(antenna_coords, CAPSULE_COORDINATES)


def two_proximate_sine_waves_experiment(antiphase=False):
    sr, signal = create_simple_sine_wave(22050, 1200, 1, 1)
    sr, signal2 = create_simple_sine_wave(22050, 1240, 1, 1)
    combination = signal + signal2
    combination = combination / max(combination)
    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
        combination[:2205], CAPSULE_COORDINATES)
    db_specs = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
    db_specs_orig = calculate_db_spec_for_time_series(combination[:2205], sr, zero_scale=True)
    if antiphase:
        antiphase_prong_plot(pointwise_time_series[0], pointwise_time_series[26])
    create_slider_for_time_series_and_db_spec(pointwise_time_series, combination[:2205], db_specs, db_specs_orig)
    animate_antenna_movement(antenna_coords, CAPSULE_COORDINATES)


def intensity_of_nonlinearities_depends_on_input_intensity_experiment():

    sr, signal = create_simple_sine_wave(22050, 1200, 1, 1)
    sr, signal2 = create_simple_sine_wave(22050, 1200, 1, 2)

    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
        signal[:220], CAPSULE_COORDINATES)
    time_series2, pointwise_time_series2, antenna_coords2 = create_time_series_and_antenna_coordinates(
        signal2[:220], CAPSULE_COORDINATES)
    db_specs = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=False, antenna=True)
    db_specs_2 = calculate_db_spec_for_time_series(pointwise_time_series2, sr, zero_scale=False, antenna=True)
    magnitude_of_nonlinearities_plot(db_specs, db_specs_2)


def paper_color_plot(sampling_rate: int, samples: int, f0_1: int, f0_2: int, secs: int) -> None:
    """
    Creates a plot for the paper which depicts the male and female mosquito sounds of a simulated synchronization with
    different colors.
    :param sampling_rate:
    :param samples: how many samples to look at
    :param f0_1: fundamental frequency of male mosquito
    :param f0_2: fundamental frequency of female mosquito
    :param secs:
    """
    sr, SIGNAL = create_mosquito_sound(sampling_rate, f0_1, secs, 1)  # Our mosquito
    _, SIGNAL1 = create_mosquito_sound(sampling_rate, f0_2, secs, 1)  # External sound source

    SIGNAL_D_1 = SIGNAL.copy() + SIGNAL1
    SIGNAL_D_1 = SIGNAL_D_1 / 2

    time_series, pointwise_time_series, antenna_coords = create_time_series_and_antenna_coordinates(
        SIGNAL_D_1[:samples], CAPSULE_COORDINATES)
    output_spec = calculate_db_spec_for_time_series(pointwise_time_series, sr, zero_scale=True, antenna=True)
    input_spec = calculate_db_spec_for_time_series(SIGNAL_D_1[:samples], sr, zero_scale=True)
    m2_spec = calculate_db_spec_for_time_series(SIGNAL1[:samples], sr, zero_scale=True)
    m1_spec = calculate_db_spec_for_time_series(SIGNAL[:samples], sr, zero_scale=True)

    create_multicolor_plot_for_paper(m1_spec, m2_spec, input_spec, output_spec)


if __name__ == "__main__":
    # Measure how long the execution takes
    start = perf_counter()

    # Create Capsule
    # 0.001 is added because arange is not inclusive
    X = np.arange(3 * pi / 4, 9 * pi / 4 + 0.001,
                  3 * pi / 2 / 79)  # Fixed base-points for the capsule, in cartesian coordinates
    # Create points for capsule in parametric form
    CAPSULE_X, CAPSULE_Y = calculate_points(X, capsule_func, capsule_func_2)
    CAPSULE_COORDINATES = np.vstack((CAPSULE_X, CAPSULE_Y)).T

    # HERE any experiment one wishes to execute should be called
    # single_sine_wave_experiment()
    # two_proximate_sine_waves_experiment(antiphase=False)
    # intensity_of_nonlinearities_depends_on_input_intensity_experiment()
    # inspect_synch_mosquito_audio("neue_synchs/sync4bp_e.wav", scale= 7, cutoff= 0.5, sr=None)  # "neue_synchs/sync2.wav"
    # paper_color_plot(22050, 2205, 600, 400, 1)
    localization([30, 60, 90], 22050, 11025, 636, 400, 1)

    # Print execution time
    stop = perf_counter()
    print(round(stop-start, 3))