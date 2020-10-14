# -*- coding: utf-8 -*- 
from flask import Flask,render_template,request,redirect
import os
import datetime
app = Flask(__name__)
DIR = "static/img"

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




@app.route('/',methods=['GET','POST'])
def check_img():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/check',methods=['GET','POST'])
def check_answer():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        img = request.files['img']
        img.save(os.path.join(DIR,img.filename))
        name = os.path.join(DIR,img.filename)
        check = img_predict(name)
        print(check)
        name_list = []
        for i in range(len(searchNameList)):
            face_per = searchNameList[i] + "顔である確率は" + str(check[i])
            name_list.append(face_per)
        os.remove(name)
        print(name_list)
        return render_template('check.html',name_list=name_list)
        



if __name__ == "__main__":
    app.run(debug=True)


