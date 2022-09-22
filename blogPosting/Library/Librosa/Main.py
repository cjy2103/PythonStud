import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 파일로딩 + 음파 분석
y, sr = librosa.load(librosa.ex('trumpet'))

# y, sr = librosa.load('test.wav')

plt.figure(figsize=(10, 5))

librosa.display.waveshow(y, sr=sr)
plt.show()
