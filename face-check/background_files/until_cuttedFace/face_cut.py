# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
from pykakasi import kakasi

# OpenCVのデフォルトの分類器のpath。(https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xmlのファイルを使う)
cascade_path = 'cutFace_cascade/haarcascade_frontalface_default.xml'
# 例
#cascade_path = './opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)



# 顔認識する対象を決定（検索ワードを入力）
SearchName = os.listdir("Original/")
ImgNumber =100
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize=(250,250)
input_shape=(250,250,3)





for name in SearchName:   
    # 画像データのあるディレクトリ
    # name = exchange_word(jp_name)
    input_data_path = "Original/" + str(name)
    check_path = str(name)
    pass_path = os.listdir("Face/")
    if not check_path in pass_path:
        # 切り抜いた画像の保存先ディレクトリを作成
        os.makedirs("Face/"+str(name), exist_ok=True)
        save_path = "Face/"+str(name) + "/"
        # 収集した画像の枚数(任意で変更)
        image_count = ImgNumber
        # 顔検知に成功した数(デフォルトで0を指定)
        face_detect_count = 0

        print("{}の顔を検出し切り取りを開始します。".format(name))

        # files = os.listdir("Original/石原さとみ")
        # 集めた画像データから顔が検知されたら、切り取り、保存する。
        for i in range(image_count):
            img = cv2.imread(input_data_path + '/'+ str(i) + '.jpg', cv2.IMREAD_COLOR)
            if img is None:
                print('image' + str(i) + ':NoFace')
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = faceCascade.detectMultiScale(gray, 1.1, 3)
                if len(face) > 0:
                    for rect in face:
                        # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
                        # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
                        # cv2.imwrite('detected.jpg', img)
                        x = rect[0]
                        y = rect[1]
                        w = rect[2]
                        h = rect[3]
                        cv2.imwrite(save_path + 'cutted' + str(face_detect_count) + '.jpg',img[y:y+h,  x:x+w])
                        face_detect_count = face_detect_count + 1
                else:
                    print('image' + str(i) + ':NoFace') 
    else:
        print(str(name) + "の顔の切り取りはすでに行っております。")

print("顔画像の切り取り作業、正常に動作しました。")