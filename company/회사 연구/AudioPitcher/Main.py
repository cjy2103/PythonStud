import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa, librosa.display
from ipywidgets import interact, fixed, interact_manual
import ipywidgets as widgets

"""
# 파일로딩
filename = 'C:/Users/YallaFactory/Desktop/abc/test.m4a'
path = os.fspath(filename)
print(os.path.exists(filename))

y, sr = librosa.load(filename, duration=10)
"""

# 'C:/Users/YallaFactory/Desktop/test/Your NameKimi no Na wa君の名は。 Orchestra Concert Nandemonaiyaなんでもないや (Movie and Credit Versions).mp3'
# 재영의 작업공간/일본어
# C:\Users\YallaFactory\Desktop\test
# C:\Users\YallaFactory\Desktop\abc
# y, sr = librosa.load(path, sr=32000)


"""
용어 설명
y :  librosa.load로 음성 데이터를 load 하여 얻은 값 (높이).
sr : sampling rate ( librosa에서 별도로 설정하지 않으면 default로 22500으로 지정됨
"""

# 파일로딩 + 음파 분석
y, sr = librosa.load(librosa.ex('trumpet'))

audio_sample_len = y.shape[-1]
print(f'#N Samples:{audio_sample_len}')
audio_len_sec = audio_sample_len / sr
print(f'Length in Sec: {audio_len_sec}')

plt.plot(y)
plt.xticks(librosa.time_to_samples(np.arange(6)), np.arange(6))
plt.xlim(librosa.time_to_samples((0, 5)))
ipd.Audio(y, rate=sr)
plt.show()

# 스펙트로그램
"""
용어 설명
S : librosa.stft 얻은값. 즉 stft(Short-Time Fourier Transform)을 하여 얻어진 magnitude(규모)와 phase(위상) 값 
"""

S = librosa.stft(y)
# imshow -> 이미지맵 (원하는 사이즈의 픽셀을 원하는 색으로 채워서 만든 그림)
plt.imshow(librosa.amplitude_to_db(S, ref=np.max), origin='lower', aspect='auto')
plt.colorbar()
plt.show()


# Mel-Spectrogram
"""
용어 설명
# plt 관련 변수
fig : 팔레트
ax : 그림하나하나

# librosa 관련 변수
y : mel-spectrogram을 얻기위해 librosa.load로 음성 데이터를 load 하여 얻은 값.
sr : sampling rate ( librosa에서 별도로 설정하지 않으면 default로 22500으로 지정됨
n_fft : time-magnitude domain 을 frequency로 바꿀때 사용됨 -> 음성의 길이를 얼마만큼 자를것인지.
hop_length : 음성의 magnitude를 얼만큼 겹친 상태로 잘라서 칸으로 보여줄것인지 정하는 변수
"""

def draw_spec(y, n_fft=2048, hop_length=None):
    fig, ax = plt.subplots()
    S = librosa.stft(y, n_fft, hop_length)
    img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                           ref=np.max),
                                   hop_length=hop_length,
                                   y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


interact(draw_spec, y=fixed(y),
         n_fft=widgets.IntSlider(min=256, max=8192, step=8, value=2048),
         hop_length=widgets.IntSlider(min=64, max=1024, step=8, value=512))


"""
용어설명
n_mels : 주파수 해상도
fmax : float 최대값
"""
def draw_mel(y, n_mels=128, n_fft=2048, hop_length=None):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length,
                                       fmax=8000)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   hop_length=hop_length,
                                   y_axis='mel', x_axis='time', fmax=8000,
                                   ax=ax)
    ax.set_title('Mel-frequency spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")


interact(draw_mel, y=fixed(y),
         n_mels=widgets.IntSlider(min=40, max=256, step=8, value=128),
         n_fft=widgets.IntSlider(min=256, max=8192, step=8, value=2048),
         hop_length=widgets.IntSlider(min=64, max=1024, step=8, value=512))
plt.show()


# 주파수 세기
"""
용어설명
rms = root mean square(제곱평균제곱근)
semilogy = 반대수 그래프 (세미 로그 그래프)

"""
def draw_power_db(y, frame_length_sec=0.01):
    fig, ax = plt.subplots(1, figsize=(8, 3))
    frame_length = librosa.time_to_samples(frame_length_sec, sr=sr)
    hop_length = 512  # 음성의 magnitude
    rms = librosa.feature.rms(y,
                              frame_length=frame_length,
                              hop_length=hop_length
                              ).T
    ax.semilogy(rms)
    # x축 제한 설정
    ax.set_xlim(
        librosa.time_to_frames(
            (0, 5),
            sr=sr,
            hop_length=hop_length)
    )
    # 눈금 설정
    ax.set_xticks(
        librosa.time_to_frames(
            np.arange(6),
            sr=sr,
            hop_length=hop_length),
        np.arange(6)
    )


plt.figure(figsize=(8, 2))
plt.plot(y)
plt.xticks(librosa.time_to_samples(np.arange(6)), np.arange(6))
plt.xlim(librosa.time_to_samples((0, 5)))
plt.show()

interact(draw_power_db, y=fixed(y),
         frame_length_sec=widgets.FloatSlider(min=0.001, max=1.0, step=0.01, value=0.1, readout_format='.3f', ))
plt.show()


# F0 Estimation (다중 그래프)
"""
용어설명 
f0 : fundamental frequency (기본 주파수)
fmin : 최소 주파수 (C2 : 65HZ 의미) 
fmax : 최대 주파수 (C7 : 2093HZ 의미) 
HZ 단위에 대해서는 : https://pages.mtu.edu/~suits/notefreqs.html 에 나와 있음

gs : GridSpec : 여러 그래프 같이 그리게 해줌 (3행 1열)
subplots() : 여러개의 그래프 그리게 해줌
"""

f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 1, hspace=.5)
axs = gs.subplots()
axs[0].plot(y)
axs[0].set_title('waveform')
axs[0].set_xticks(librosa.time_to_samples(np.arange(6)), np.arange(6))
axs[0].set_xlim(librosa.time_to_samples((0, 5)))

axs[1].plot(f0)
axs[1].set_title('F0')
axs[1].set_xlim(librosa.time_to_frames((0,5),sr=sr))
axs[1].set_xticks(librosa.time_to_frames(np.arange(6), sr=sr), np.arange(6))

axs[2].plot(voiced_probs)
axs[2].set_title('Voice Prob')
axs[2].set_xlim(librosa.time_to_frames((0,5),sr=sr))
axs[2].set_xticks(librosa.time_to_frames(np.arange(6), sr=sr), np.arange(6))
plt.show()

# F0 Estimation (스펙 트럼)
times = librosa.times_like(f0)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, axs = plt.subplots(2, figsize=(8, 8))
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=axs[0])
axs[0].set_title('pYIN fundamental frequency estimation')
fig.colorbar(img, ax=axs[0], format="%+2.f dB")
axs[0].plot(times, f0, label='f0', color='cyan', linewidth=3)
axs[0].legend(loc='upper right')

img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=axs[1])
axs[1].set_title('pYIN fundamental frequency estimation')
fig.colorbar(img, ax=axs[1], format="%+2.f dB")
axs[1].plot(times, f0, label='f0', color='cyan', linewidth=3, alpha=0.8)
axs[1].legend(loc='upper right')
axs[1].set_xlim(0.5, 2)
ipd.Audio(y, rate=sr)
plt.show()

# MFCC ( Mel Frequency Cepstral Coefficient )
"""
용어설명 
MFCC : Mel Spectrogram을 DCT(Discrete Consine Transform) 처리하면 얻게되는 coefficient
-> mel scale로 변환한 스펙트로그램을 더 적은 값들로 압축하는 과정
DCT : Discrete Cosine Transform : DFT와 유사하지만 기저함수로 복소 지수함수가 아닌 코사인 함수를 사용한다. DFT보다 계산이 간단하고 실수만 출력한다는 장점이 있어서 DFT 대용으로 많이 사용된다.
"""

mfcc = librosa.feature.mfcc(y)
fig, ax = plt.subplots()
img = ax.imshow(mfcc, aspect='auto', origin='lower')
ax.set_xlim(librosa.time_to_frames((0,5),sr=sr))
ax.set_xticks(librosa.time_to_frames(np.arange(6), sr=sr), np.arange(6))
fig.colorbar(img, ax=ax)
plt.show()

# 위쪽의 MFCC 변화를 그래프로 표현 ( [0.5, 1, 1.5, 4]초 에서 MFCC 변화 )
fig, axs = plt.subplots(4)
axs[0].plot(mfcc[:,librosa.time_to_frames(0.5)])
axs[1].plot(mfcc[:,librosa.time_to_frames(1)])
axs[2].plot(mfcc[:,librosa.time_to_frames(1.5)])
axs[3].plot(mfcc[:,librosa.time_to_frames(4)])
plt.show()

# Spectral Centroid
"""
용어설명 
Spectral Centroid : 스펙트럼 중심은 스펙트럼을 특성화하기 위해 디지털 신호 처리에 사용되는 측정을 의미함
 스펙트럼의 질량 중심이 있는 위치를 나타냄
"""

centroid = librosa.feature.spectral_centroid(y).T
fig, ax = plt.subplots()
ax.plot(centroid)
ax.set_xlim(librosa.time_to_frames((0,5),sr=sr))
ax.set_xticks(librosa.time_to_frames(np.arange(6), sr=sr), np.arange(6))
plt.show()