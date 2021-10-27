from math import pi, cos, sin

import numpy as np


def antenna_func(v):
    return abs(cos(v)) * cos(v)


def antenna_func_2(v):
    return abs(cos(v)) * sin(v)


def capsule_func(v):
    return (0.2 + abs(cos(v))) * cos(v)


def capsule_func_2(v):
    return (0.2 + abs(cos(v))) * sin(v)


def calculate_points(x, func_x, func_y):
    points_x = np.array(list(map(func_x, x)))
    points_y = np.array(list(map(func_y, x)))

    return points_x, points_y


def create_simple_sine_wave(sr, freq, duration, mult):
    # Use this to create sample sine waves as test input for the model
    # The signal should be normalized to only contain values between -1 and 1
    signal = (mult*np.sin(2 * np.pi * np.arange(sr * duration) * freq / sr)).astype(np.float32)

    return sr, signal


def create_mosquito_sound(sr, f0, duration, mult):
    sampling_rate, signal = create_simple_sine_wave(sr, f0, duration, mult)
    _, signal1 = create_simple_sine_wave(sr, f0*2, duration, mult*0.95)  # Factors are based on real mosquito relations between harmonics
    _, signal2 = create_simple_sine_wave(sr, f0*3, duration, mult*0.8)
    m_signal = signal + signal1 + signal2
    return sampling_rate, m_signal


def move_antenna(signal_sample):
    # 0.001 is added because arange is not inclusive
    shifted_x = np.arange(3 * pi / 4 - pi / 30 * signal_sample, 9 * pi / 4 - pi / 30 * signal_sample + 0.001, 3 * pi / 2 / 79)
    antenna_x, antenna_y = calculate_points(shifted_x, antenna_func, antenna_func_2)

    return antenna_x, antenna_y


def create_time_series_and_antenna_coordinates(signal, capsule_coordinates):
    # We have 80 points describing the antenna and capsule and discard 40 because of the symmetry
    signal_distances = np.zeros(shape=(signal.size, 40))
    antenna_coordinates = []

    for i, sample in np.ndenumerate(signal):
        current_antenna_x, current_antenna_y = move_antenna(sample)
        current_antenna_coordinates = np.vstack((current_antenna_x, current_antenna_y)).T
        antenna_coordinates.append(current_antenna_coordinates)

        current_distances = np.linalg.norm(current_antenna_coordinates-capsule_coordinates, axis=1)
        current_distances = current_distances[:current_distances.size//2]
        # current_distances = abs(0.2-current_distances)  # 0.2 is the initial euclidean difference between capsule and antenna
        signal_distances[i] = current_distances

    pointwise_antenna_response = signal_distances.T

    return signal_distances, pointwise_antenna_response, antenna_coordinates


def calculate_db_spec_for_time_series(time_series, sr, zero_scale=True, antenna=False):
    if time_series.ndim < 2:
        time_series = np.array([time_series])
    db_specs = []
    for signal in time_series:
        N = signal.size

        sp = np.fft.rfft(signal)  # spectrum

        freq = np.arange((N / 2) + 1) / (float(N) / sr)  # frequencies

        # Scale by two, because we only take first half of spectrum
        s_mag = np.abs(sp) * 2
        s_mag[0] = 0

        # Convert to dBFS
        if zero_scale:
            s_dbfs = 20 * np.log10(s_mag / max(s_mag))
            if antenna:
                s_dbfs = 20 * np.log10(s_mag / max(s_mag[1:]))
        else:
            s_dbfs = 20 * np.log10(s_mag)

        # s_dbfs = s_mag / abs(max(s_mag))  # Used for mean calculation

        db_specs.append(np.array([freq, s_dbfs]))

    return db_specs


def save_specific_frequencies(db_specs, antenna_no, degrees, frequencies, factor):
    """

    :param db_specs:
    :param antenna_no:
    :param degrees: List of degrees used in localization
    :param frequencies: Frequencies one wishes to save
    :param factor: size of the frequency bins
    :return:
    """
    relevant_freqs = []
    for db_index in range(40):
        prong = []
        for f in frequencies:
            prong.append(db_specs[db_index][1][f//factor])
        relevant_freqs.append(prong)

    np.save(f"a{antenna_no}_{degrees}", relevant_freqs)