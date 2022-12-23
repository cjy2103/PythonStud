import socket
import requests
import re

#내부 IP 주소 확인 (일부 환경에 의해 틀릴수 있음 ex.VM 환경)
in_addr = socket.gethostbyname(socket.gethostname())
print(in_addr)

#외부 사이트 접속을 이용해서 내부 IP주소 알아내기 -> 더 정확
in_addr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
in_addr.connect(("www.naver.com",443))
print(in_addr.getsockname()[0])

#네이버 IP주소 알아내기
naver = "www.naver.com"
naver_ip = socket.gethostbyname(naver)
print(naver_ip)

#외부 IP 알아내기
req = requests.get("http://ipconfig.kr")
out_addr = re.search(r"IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", req.text)[1]

print(out_addr)
