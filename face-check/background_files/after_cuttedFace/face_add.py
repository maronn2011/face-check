import os
import cv2
import glob
from scipy import ndimage
import cv2
import numpy as np

# OpenCVのデフォルトの分類器のpath。(https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xmlのファイルを使う)
cascade_path = 'haarcascade_frontalface_default.xml'
# 例
#cascade_path = './opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)




# 顔認識する対象を決定（検索ワードを入力）
SearchName = os.listdir("Face/")
ImgNumber =100
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize=(250,250)
input_shape=(250,250,3)
"""
Faceディレクトリから画像を読み込んで回転、ぼかし、閾値処理をしてFaceEditedディレクトリに保存する.
"""
for name in SearchName:
    check_path = str(name)
    pass_path = os.listdir("FaceEdited/")
    if not check_path in pass_path:
        print("{}の写真を増やします。".format(name))
        in_dir = "Face/"+name+"/*"
        out_dir = "FaceEdited/"+name
        os.makedirs(out_dir, exist_ok=True)
        in_jpg=glob.glob(in_dir)
        img_file_name_list=os.listdir("Face/"+name+"/")
        for i in range(len(in_jpg)):
            #print(str(in_jpg[i]))
            img = cv2.imread(str(in_jpg[i]))
            # 回転
            for ang in [-10,0,10]:
                img_rot = ndimage.rotate(img,ang)
                img_rot = cv2.resize(img_rot,ImgSize)
                fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+".jpg")
                cv2.imwrite(str(fileName),img_rot)
                # 閾値
                img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
                fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"thr.jpg")
                cv2.imwrite(str(fileName),img_thr)
                # ぼかし
                img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
                fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"filter.jpg")
                cv2.imwrite(str(fileName),img_filter)
        print(name + "の画像の水増しに大成功しました！")
    else:
        print(name + "の画像の水増しはすでに行っております")
