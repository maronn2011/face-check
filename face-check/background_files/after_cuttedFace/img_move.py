# 2割をテストデータに移行
import shutil
import random
import glob
import os



# 顔認識する対象を決定（検索ワードを入力）
SearchName = os.listdir("FaceEdited/")

for name in SearchName:
    check_path = str(name)
    pass_path = os.listdir("test/")
    if not check_path in pass_path:
        in_dir = "FaceEdited/"+name+"/*"
        in_jpg=glob.glob(in_dir)
        img_file_name_list=os.listdir("FaceEdited/"+name+"/")
        #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
        random.shuffle(in_jpg)
        os.makedirs('test/' + name, exist_ok=True)
        for t in range(len(in_jpg)//5):
            shutil.move(str(in_jpg[t]), "test/"+name)
    else:
        print(name + "の画像はすでに移動しています")