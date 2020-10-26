# -*- coding: utf-8 -*- 
import cv2
import numpy as np
import os
from flask import Flask,render_template,request,redirect
import datetime
app = Flask(__name__)
DIR = "static/img/"

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# 対象の人の名前を取得
# searchNameList = os.listdir("../test/")
# 名前を日本語で表示させたいときはtestフォルダの順番通りに名前をリストで記述する
searchNameList = ["石原さとみ","佐々木のぞみ"]

# load_model()内は試したい学習モデルのパスを入れる
vgg_model = load_model('../results/vgg16_Final.h5')
img_width, img_height = 150, 150

# 画像を読み込んで予測する
def img_predict(filename):
    # 画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    # これを忘れると結果がおかしくなるので注意
    x = x / 255.0   


    #画像の人物を予測    
    pred = vgg_model.predict(x)[0]
    pred = pred*100
    #結果を表示する
    return pred

# OpenCVのデフォルトの分類器のpath。(https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xmlのファイルを使う)
cascade_path = '../cutFace_cascade/haarcascade_frontalface_default.xml'
# 例
#cascade_path = './opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)



# 顔認識する対象を決定（検索ワードを入力）
SearchName = os.listdir("../Original/")
ImgNumber =100
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize=(250,250)
input_shape=(250,250,3)



# 画像内に顔の切り出しを行う関数
def face_cut(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    if len(face) > 0:
        for rect in face:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            cv2.imwrite(filename,img[y:y+h,  x:x+w])
    else:
        print('image' + ':NoFace') 

@app.route('/',methods=['GET','POST'])
def check_img():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/check',methods=['GET','POST'])
def check_answer():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # フォームからファイル情報を取得
        img = request.files['img']
        # 画像を保存し、保存した画像の名前を取得
        img.save(os.path.join(DIR,img.filename))
        name = os.path.join(DIR,img.filename)
        # フォームからの画像を顔部分がある場合切り出しを行う
        face_cut(name)
        # 学習済みモデルに読み込ませ一致具合の判定を行う
        check = img_predict(name)
        name_list = []
        # 結果の出力
        for i in range(len(searchNameList)):
            face_per = searchNameList[i] + "顔である確率は" + str(check[i])
            name_list.append(face_per)
        return render_template('check.html',name_list=name_list)
        



if __name__ == "__main__":
    app.run(debug=True)


