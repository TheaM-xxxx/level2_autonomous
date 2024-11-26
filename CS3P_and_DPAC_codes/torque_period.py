# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, medfilt

list1 = [717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 778, 779, 780, 781, 782, 783, 784, 786, 786, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797]
list2 = [11574, 11594, 11598, 11585, 11649, 11675, 11626, 11647, 11660, 11641, 11632, 11639, 11620, 11667, 11608, 11632, 11612, 11649, 11606, 11594, 11585, 11555, 11620, 11622, 11545, 11520, 11542, 11551, 11557, 11516, 11491, 11491, 11481, 11487, 11484, 11476, 11432, 11404, 11429, 11455, 11453, 11451, 11382, 11393, 11420, 11469, 11499, 11473, 11582, 11556, 11583, 11590, 11571, 11588, 11626, 11624, 11618, 11637, 11664, 11705, 11682, 11654, 11685, 11731, 11749, 11770, 11744, 11718, 11709, 11735, 11772, 11818, 11739, 11782, 11754, 11804, 11638, 11772, 11780, 11781, 11781, 11780, 11784, 11784, 11787, 11792, 11795, 11799, 11799, 11805, 11797, 11799, 11800, 11800, 11803, 11805, 11803, 11806, 11799, 11801]
def rescale_sequence(sequence, target_length):
    original_length = len(sequence)
    if original_length == target_length:
        return sequence

    scaled_sequence = []
    scale_factor = target_length / original_length

    for i in range(target_length):
        index = int(i / scale_factor)
        if index >= original_length:
            index = original_length - 1
        scaled_sequence.append(sequence[index])

    return scaled_sequence

def average_lists(lists):
    # Get the length of the list, assuming all lists are the same length.
    n = len(lists[0])
    # Create an empty list to store averages
    avg = []
    # Traverse each position
    for i in range(n):
        # Calculate the sum of the elements at that position
        total = 0
        for lst in lists:
            total += lst[i]
        # Calculate the average for the location and add it to the list of averages
        avg.append(total / len(lists))
    # Returns a list of averages
    return avg


def period_analyse(data_input):
    # Median filtering
    torque_signal = medfilt(data_input[1])

    # Wavelet transform
    wavelet = 'db4'  # Wavelet types
    level = 5  # Decomposition levels
    coeffs = pywt.wavedec(torque_signal, wavelet, level=level)

    # Reconstruct coefficients at level
    reconstructed_signal = pywt.waverec(coeffs[:level] + [None] * (len(coeffs) - level), wavelet)

    # Cycle identification
    peaks, _ = find_peaks(reconstructed_signal, distance=80)  # Recognize cycles through peak detection

    last_peak = 0
    list_period = []
    list_period_data = []
    rescale_list_period_data = []
    # Draw a red vertical line to cut each cycle
    for peak in peaks:
        plt.axvline(x=peak, color='red', linestyle='--')
        list_period.append(peak - last_peak)
        last_peak = peak

    print(peaks)

    list_period = list_period[1:]  # Remove the first incomplete cycle
    avg_period = sum(list_period) / len(list_period)
    print('avg_period:', avg_period)

    # Adjust the data for each cycle to an average length
    for i in range(len(peaks) - 1):
        res = rescale_sequence(torque_signal[peaks[i]:peaks[i + 1]], round(avg_period))
        rescale_list_period_data.append(res)
        list_period_data.append(torque_signal[peaks[i]:peaks[i + 1]])

    # Mechanical cycle noise
    avg_list = average_lists(rescale_list_period_data)

    # Phase bias
    phase_bias = len(data_input[0]) - max(peaks)

    return avg_list, phase_bias

def Torque_data_interpolation(list_Displacement_data, list_Torque_data):
    # Get the unique value and the corresponding index
    unique_data2, unique_indices = np.unique(list_Displacement_data, return_inverse=True)

    # Calculate the average data1 value of duplicate values in data2
    mean_data1 = np.zeros_like(unique_data2)
    for i, val in enumerate(unique_data2):
        list_Torque_data = np.array(list_Torque_data)  # Convert lists to NumPy arrays

        mean_data1[i] = np.mean(list_Torque_data[np.where(unique_indices == i)])

    # Generate a range of consecutive integers data2
    new_data2 = np.arange(np.min(list_Displacement_data), np.max(list_Displacement_data) + 1)

    # Perform linear interpolation
    interpolated_data1 = np.interp(new_data2, unique_data2, mean_data1)



    return new_data2, interpolated_data1


