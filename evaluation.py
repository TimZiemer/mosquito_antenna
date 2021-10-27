import os

import matplotlib.pyplot as plt
import numpy as np

"""
This file can be used to evaluate the localization experiment (see mosquito_antenna.py).
The paths in the functions might have to be adapted.
In the main section the frequencies and degrees might have to be adapted, depending on how you conduct the experiment.
"""


def create_barplots(degrees, freqs):
    for deg in degrees:
        a1 = np.load(f"a1_{deg}.npy")
        a2 = np.load(f"a2_{deg}.npy")
        antennas = ['Antenna 1','Equal', 'Antenna 2 (further away)']

        for j, freq in enumerate(freqs):
            print(freq)
            a1_greater_prongs = []
            f1 = [0, 0, 0]
            for i, (p1, p2) in enumerate(zip(a1, a2)):
                if p1[j] > p2[j]:
                    a1_greater_prongs.append(i+1)
                    f1[0] += 1
                elif p1[j] == p2[j]:
                    f1[1] += 1
                else:
                    f1[2] += 1
            print(a1_greater_prongs)
            fig = plt.figure()
            plt.bar(antennas, f1)
            plt.text(0, 1, str(f1[0]), fontweight='bold')
            plt.text(1, 2, str(f1[1]), fontweight='bold')
            plt.text(2, 3, str(f1[2]), fontweight='bold')
            plt.title(f"{deg} Degree angle; {freq} Hz: \n Number of Prongs where one value is larger than the other")
            plt.ylabel("Prongs")
            plt.savefig(f"{deg}_degree_{freq}_hz.png")
            plt.close()


def calc_mean_etc(degrees):
    for deg in degrees:
        print(deg)
        a1 = np.load(f"linear/a1_{deg}.npy")
        a2 = np.load(f"linear/a2_{deg}.npy")
        with open(f"means_{deg}.txt", "w") as f:
            f.write("A1:\n")
            f.write("\t72 Hz 164 Hz 236 Hz\n")
            f.write("\tMean " + str(np.mean(a1, axis=0)) + "\n")
            f.write("\tSTD " + str(np.std(a1, axis=0)) + "\n")

            f.write("A2:\n")
            f.write("\t72 Hz 164 Hz 236 Hz\n")
            f.write("\tMean " + str(np.mean(a2, axis=0)) + "\n")
            f.write("\tSTD " + str(np.std(a2, axis=0)) + "\n")


def calc_db_differences(degrees, freqs):
    for deg in degrees:
        print(deg)
        a1 = np.load(f"a1_{deg}.npy")
        a2 = np.load(f"a2_{deg}.npy")

        try:
            os.remove(f"differences_{deg}.txt")
        except OSError:
            pass

        for i, freq in enumerate(freqs):
            ratio = a1[:,i] / a2[:,i]

            # Mean and range differences
            normratio = np.mean(ratio)
            realdifference_dB = 20 * np.log10(normratio)

            min_ratio, max_ratio = np.min(ratio), np.max(ratio)
            amin_ratio, amax_ratio = np.argmin(ratio), np.argmax(ratio)
            min_ratio_dB, max_ratio_dB = 20 * np.log10(min_ratio), 20 * np.log10(max_ratio)

            print(amin_ratio, amax_ratio)
            np.save(f"diffs_{deg}_{freq}.npy", realdifference_dB)

            with open(f"differences_{deg}.txt", "a") as f:
                f.write(f"{freq} Hz:\n")
                f.write("\tMean " + str(realdifference_dB) + "\n")
                f.write("\tMean Lin " + str(normratio) + "\n")
                f.write("\tMin " + str(min_ratio_dB) + "\n")
                f.write("\tMin Lin " + str(min_ratio) + "\n")
                f.write("\tMax " + str(max_ratio_dB) + "\n")
                f.write("\tMax Lin " + str(max_ratio) + "\n")

if __name__ == "__main__":
    freqs = [72, 164, 236]
    degrees = [30, 60, 90]
    #calc_mean_etc(degrees)
    create_barplots(degrees, freqs)
    calc_db_differences(degrees, freqs)
