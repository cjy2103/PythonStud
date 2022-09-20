import os
import sys
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy.interpolate
import libfmp.c2
from libfmp.b.b_plot import plot_signal, plot_chromagram
from libfmp.c3.c3s2_dtw_plot import plot_matrix_with_points

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic, evaluate_synchronized_positions
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning


Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6

figsize = (9, 3)

print(sys.path)

# sys.path.append('C:/Users/YallaFactory/Desktop/PythonChord/company/AudioSort/AudioSort/synctoolbox/synctoolbox')

# Loading two recordings of the same piece
audio_1, _ = librosa.load('/synctoolbox/data_music/Schubert_D911-03_HU33.wav', Fs)


plot_signal(audio_1, Fs=Fs, ylabel='Amplitude', title='Version 1', figsize=figsize)
plt.show()

audio_2, _ = librosa.load('/synctoolbox/data_music/Schubert_D911-03_SC06.wav', Fs)

plot_signal(audio_2, Fs=Fs, ylabel='Amplitude', title='Version 2', figsize=figsize)
plt.show()
ipd.display(ipd.Audio(audio_2, rate=Fs))

# Estimating tuning
tuning_offset_1 = estimate_tuning(audio_1, Fs)
tuning_offset_2 = estimate_tuning(audio_2, Fs)
print('Estimated tuning deviation for recording 1: %d cents, for recording 2: %d cents' % (tuning_offset_1, tuning_offset_2))

def get_features_from_audio(audio, tuning_offset, visualize=True):
    f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate, verbose=visualize)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate, feature_sequence_length=f_chroma_quantized.shape[1], visualize=visualize)
    return f_chroma_quantized, f_DLNCO


f_chroma_quantized_1, f_DLNCO_1 = get_features_from_audio(audio_1, tuning_offset_1)
f_chroma_quantized_2, f_DLNCO_2 = get_features_from_audio(audio_2, tuning_offset_2)



# The next plots illustrate the different representations of the first 30 seconds of each version.
#
# plot_chromagram(f_chroma_quantized_1[:, :30 * feature_rate], Fs=feature_rate, title='Chroma representation for version 1', figsize=figsize)
# plt.show()
# plot_chromagram(f_DLNCO_1[:, :30 * feature_rate], Fs=feature_rate, title='DLNCO representation for version 1', figsize=figsize)
# plt.show()
#
# plot_chromagram(f_chroma_quantized_2[:, :30 * feature_rate], Fs=feature_rate, title='Chroma representation for version 2', figsize=figsize)
# plt.show()
# plot_chromagram(f_DLNCO_2[:, :30 * feature_rate], Fs=feature_rate, title='DLNCO representation for version 2', figsize=figsize)
# plt.show()


# Finding optimal shift of chroma vectors ( Error 발생하면서 안돌아감 )
f_cens_1hz_1 = quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, feature_rate)[0]
f_cens_1hz_2 = quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, feature_rate)[0]
opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)
print('Pitch shift between recording 1 and recording 2, determined by DTW:', opt_chroma_shift, 'bins')

f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

plot_chromagram(f_chroma_quantized_1[:, :30 * feature_rate], Fs=feature_rate, title='Version 1', figsize=figsize)
plt.show()
plot_chromagram(f_chroma_quantized_2[:, :30 * feature_rate], Fs=feature_rate, title='Version 2, shifted to match version 1', figsize=figsize)
plt.show()

# wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_1, f_onset1=f_DLNCO_1, f_chroma2=f_chroma_quantized_2, f_onset2=f_DLNCO_2, input_feature_rate=feature_rate, step_weights=step_weights, threshold_rec=threshold_rec, verbose=True)