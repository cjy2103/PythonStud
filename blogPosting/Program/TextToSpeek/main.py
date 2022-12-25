from gtts import gTTS
from playsound import playsound


text = "음성 파일 테스트 예제 입니다."

tts = gTTS(text=text, lang="ko")
tts.save(r"음성파일\test.mp3")

