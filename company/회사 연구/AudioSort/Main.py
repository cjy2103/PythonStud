import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy.interpolate
import libtsm
from libfmp.b.b_plot import plot_signal, plot_chromagram
from libfmp.c3.c3s2_dtw_plot import plot_matrix_with_points

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic, \
    evaluate_synchronized_positions
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

"""
꼭 읽으세요!!!!!!
Python 은 버전마다 라이브러리 가 호환이 안되는 경우가 매우 많습니다.
이번 예제에서 사용하는 synctoolbox 라이브 러리 역시 해당 문제가 발생했습니다.
synctoolbox 라이브러리를 직접 만드신 분이 Python 3.10 버전으로 버전업 이 되면서 Numpy 라이브러리 역시 버전이 올라갔는데
이 Numpy 버전이 해당 라이브러리와 호환이 되지 않는다고 하였습니다.
따라서 위 라이브러리를 쓰기 위해서는 Python 버전을 3.9 버전 대로 내려야 사용이 가능합니다.
해당 이슈 내용 : https://github.com/meinardmueller/synctoolbox/issues/19
"""

"""
용어설명
FS : sr (sampling rate)
feature_rate : Features per second
DTW: Dynamic time warping ( 동적 시간 워핑 ) 
step_weights : DTW step weights
threshold_rec :        
 Defines the maximum area that is spanned by the rectangle of two
 consecutive elements in the alignment (default: 10000)
"""
Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6

figsize = (9, 3)  # 그림판 크기

# 두 파일 열기
audio_1, _ = librosa.load('data_music/Schubert_D911-03_HU33.wav', Fs)

plot_signal(audio_1, Fs=Fs, ylabel='Amplitude', title='Version 1', figsize=figsize)
plt.show()

audio_2, _ = librosa.load('data_music/Schubert_D911-03_SC06.wav', Fs)

plot_signal(audio_2, Fs=Fs, ylabel='Amplitude', title='Version 2', figsize=figsize)
plt.show()
ipd.display(ipd.Audio(audio_2, rate=Fs))

"""
용어설명
f_pitch :
    Matrix containing the extracted pitch-based features ( 추출된 피치 기반 특징을 포함하는 행렬 )

f_chroma :
    Rows of 'f_pitch' between ``midi_min`` and ``midi_max``,
    aggregated into chroma bands.

f_chroma_quantized :
    Quantized chroma representation

f_pitch_onset : 
   return 
    f_peaks : dict
        A dictionary of onset peaks:
            * Each key corresponds to the midi pitch number
            * Each value f_peaks[midi_pitch] is an array of doubles of size 2xN:
                * First row give the positions of the peaks in milliseconds.
                * Second row contains the corresponding magnitudes of the peaks.
                
f_DLNCO :
    Decaying Locally adaptively Normalized Chroma Onset features
"""

# Estimating tuning (튜닝 추정)
# Computing quantized chroma and DLNCO features ( 정량화된 크로마 및 DLNCO 기능 계산 )
tuning_offset_1 = estimate_tuning(audio_1, Fs)
tuning_offset_2 = estimate_tuning(audio_2, Fs)
print('Estimated tuning deviation for recording 1: %d cents, for recording 2: %d cents' % (
    tuning_offset_1, tuning_offset_2))


def get_features_from_audio(audio, tuning_offset, visualize=True):
    f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate,
                                      verbose=visualize)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate,
                                            feature_sequence_length=f_chroma_quantized.shape[1], visualize=visualize)
    return f_chroma_quantized, f_DLNCO


f_chroma_quantized_1, f_DLNCO_1 = get_features_from_audio(audio_1, tuning_offset_1)
f_chroma_quantized_2, f_DLNCO_2 = get_features_from_audio(audio_2, tuning_offset_2)

# The next plots illustrate the different representations of the first 30 seconds of each version.
# (다음 코드는 각 버전의 처음 30초의 다른 표현을 보여 줍니다.)

plot_chromagram(f_chroma_quantized_1[:, :30 * feature_rate], Fs=feature_rate,
                title='Chroma representation for version 1', figsize=figsize)
plt.show()
plot_chromagram(f_DLNCO_1[:, :30 * feature_rate], Fs=feature_rate, title='DLNCO representation for version 1',
                figsize=figsize)
plt.show()

plot_chromagram(f_chroma_quantized_2[:, :30 * feature_rate], Fs=feature_rate,
                title='Chroma representation for version 2', figsize=figsize)
plt.show()
plot_chromagram(f_DLNCO_2[:, :30 * feature_rate], Fs=feature_rate, title='DLNCO representation for version 2',
                figsize=figsize)
plt.show()

# Finding optimal shift of chroma vectors ( 크로마 벡터의 최적 이동 찾기 )

"""
용어 설명 
f_cens_1hz : 
    return 
    f_CENS: np.ndarray
        CENS (Chroma Energy Normalized Statistics) features

    CENS_feature_rate: float
        Feature rate of the CENS features
        
opt_chroma_shift :
    Optimal chroma shift which minimizes the DTW cost. ( DTW 비용을 최소화하는 최적의 크로마 시프트.)
    
f_chroma_quantized :
   return 
   Shifted chroma representation
   
wp : 
    return 
    Resulting warping path
    
audio_1_shifted :
    return 
    The time-scale modified output signal ( 시간 스케일 수정 출력 신호 )
"""

f_cens_1hz_1 = quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, feature_rate)[0]
f_cens_1hz_2 = quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, feature_rate)[0]
opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)
print('Pitch shift between recording 1 and recording 2, determined by DTW:', opt_chroma_shift, 'bins')

f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

plot_chromagram(f_chroma_quantized_1[:, :30 * feature_rate], Fs=feature_rate, title='Version 1', figsize=figsize)
plt.show()
plot_chromagram(f_chroma_quantized_2[:, :30 * feature_rate], Fs=feature_rate,
                title='Version 2, shifted to match version 1', figsize=figsize)
plt.show()

wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_1, f_onset1=f_DLNCO_1, f_chroma2=f_chroma_quantized_2,
                      f_onset2=f_DLNCO_2, input_feature_rate=feature_rate, step_weights=step_weights,
                      threshold_rec=threshold_rec, verbose=True)

pitch_shift_for_audio_1 = -opt_chroma_shift % 12
if pitch_shift_for_audio_1 > 6:
    pitch_shift_for_audio_1 -= 12
audio_1_shifted = libtsm.pitch_shift(audio_1, pitch_shift_for_audio_1 * 100, order="tsm-res")

# The TSM functionality of the libtsm library expects the warping path to be given in audio samples.
# Here, we do the conversion and additionally clip values that are too large.
# time_map = wp.T / feature_rate * Fs
# time_map[time_map[:, 0] > len(audio_1), 0] = len(audio_1) - 1
# time_map[time_map[:, 1] > len(audio_2), 1] = len(audio_2) - 1
#
# y_hpstsm = libtsm.hps_tsm(audio_1_shifted, time_map)
# stereo_sonification = np.hstack((audio_2.reshape(-1, 1), y_hpstsm))
#
# print('Original signal 1', flush=True)
# ipd.display(ipd.Audio(audio_1, rate=Fs))
#
# print('Original signal 2', flush=True)
# ipd.display(ipd.Audio(audio_2, rate=Fs))
#
# print('Synchronized versions', flush=True)
# ipd.display(ipd.Audio(stereo_sonification.T, rate=Fs))

# Transferring measure annotations

measure_annotations_1 = pd.read_csv(filepath_or_buffer='data_csv/Schubert_D911-03_HU33.csv', delimiter=';')['start']
measure_positions_1_transferred_to_2 = scipy.interpolate.interp1d(wp[0] / feature_rate, wp[1] / feature_rate, kind='linear')(measure_annotations_1)
measure_annotations_2 = pd.read_csv(filepath_or_buffer='data_csv/Schubert_D911-03_SC06.csv', delimiter=';')['start']

mean_absolute_error, accuracy_at_tolerances = evaluate_synchronized_positions(measure_annotations_2 * 1000, measure_positions_1_transferred_to_2 * 1000)