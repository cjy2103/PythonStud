import qrcode
import os

path = os.path.dirname(os.path.abspath(__file__))
# qr_data = "www.naver.com"
# qr_image = qrcode.make(qr_data)
#
# save_path = path + r"\qrcode\네이버.png"
# qr_image.save(save_path)

qr_list = ['네이버', '블루아카이브', 'Mychord', '우마무스메', '내블로그']

file_path = path + r'\qrcode\qrcode_make_list'
pos = 0
with open(file_path, 'rt', encoding='UTF8') as f:
    read_line = f.readlines()

    for line in read_line:
        # 공백제거
        line = line.strip()

        qr_data = line
        qr_img = qrcode.make(qr_data)

        save_path = path + r'\qrcode\\' + qr_list[pos] + '.png'
        qr_img.save(save_path)
        pos = pos + 1
