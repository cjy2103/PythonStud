import string
import itertools
import zipfile

global password


def find_key(passwd_string, min_len, max_len, zFile):
    for len in range(min_len, max_len + 1):
        to_attempt = itertools.product(passwd_string, repeat=len)
        for attempt in to_attempt:
            passwd = ''.join(attempt)
            print(passwd)
            try:
                zFile.extractall(pwd=passwd.encode())
                global password
                password = passwd
                return 1
            except:
                pass


passwd_string = '0123456789' + string.ascii_letters
zFile = zipfile.ZipFile(r'secret_file.zip')

min_len = 1
max_len = 6

find_result = find_key(passwd_string, min_len, max_len, zFile)

if find_result == 1:
    print(f'암호찾기 성공: 패스워드: {password}')
else :
    print("암호찾기 실패")
