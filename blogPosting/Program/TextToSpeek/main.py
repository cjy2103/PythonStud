from gtts import gTTS
from playsound import playsound
import os

path = os.path.dirname(os.path.abspath(__file__))
text = "음성 파일 테스트 예제 입니다."

tts = gTTS(text=text, lang="ko")
tts.save(r"soundfile\test.mp3")

# playsound(path+r"\soundfile\test.mp3")

file_path = path+r"\text\text_file"

with open(file_path, "rt", encoding="UTF8") as f:
    read_file = f.read()

tfts = gTTS(text=read_file, lang="ko")
tfts.save(r"soundfile\Text.mp3")

playsound(path+r"\soundfile\Text.mp3")


