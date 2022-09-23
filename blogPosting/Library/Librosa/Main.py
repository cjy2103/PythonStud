import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from ipywidgets import interact, fixed, interact_manual
import ipywidgets as widgets

# 파일로딩 + 음파 분석
y, sr = librosa.load(librosa.ex('trumpet'))

# y, sr = librosa.load('test.wav')

plt.figure(figsize=(10, 5))

librosa.display.waveshow(y, sr=sr)
plt.show()

# Mel - Spectrogram

S = librosa.stft(y)
# imshow -> 이미지맵 (원하는 사이즈의 픽셀을 원하는 색으로 채워서 만든 그림)
plt.imshow(librosa.amplitude_to_db(S, ref=np.max), origin='lower', aspect='auto')
plt.colorbar()
plt.show()

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

# MFCC
mfcc = librosa.feature.mfcc(y)
fig, ax = plt.subplots()
img = ax.imshow(mfcc, aspect='auto', origin='lower')
ax.set_xlim(librosa.time_to_frames((0,5),sr=sr))
ax.set_xticks(librosa.time_to_frames(np.arange(6), sr=sr), np.arange(6))
fig.colorbar(img, ax=ax)
plt.show()