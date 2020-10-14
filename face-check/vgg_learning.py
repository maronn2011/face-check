import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
import time
import background_files.after_cuttedFace.face_add
import background_files.after_cuttedFace.img_move

# 分類するクラス
classes = os.listdir("./FaceEdited")
nb_classes = len(classes)
#画像の大きさを設定
img_width, img_height = 150, 150

# トレーニング用とバリデーション用の画像格納先（パスは自分で設定してください）
train_data_dir = 'FaceEdited/'
validation_data_dir = 'test/'

#トレーニングデータ用の画像数
nb_train_samples = 598
#バリデーション用の画像数
nb_validation_samples = 149
#バッチサイズ
batch_size = 8
#エポック数
nb_epoch = 10

# トレーンング用、バリデーション用データを生成するジェネレータ作成
train_datagen = ImageDataGenerator(rescale=1.0 / 255,)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)


  # VGG16のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# VGG16の図の緑色の部分（FC層）の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# VGG16とFC層を結合してモデルを作成（完成図が上の図）
vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# VGG16の図の青色の部分は重みを固定（frozen）
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# 多クラス分類を指定
vgg_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])


# Fine-tuning
history = vgg_model.fit_generator(
    train_generator,
    # steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    # validation_steps=nb_validation_samples
    )



# 学習結果を描写
import matplotlib.pyplot as plt

#acc, val_accのプロット
plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
#Final.pngという名前で、結果を保存
plt.show()


#resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# 重みを保存
vgg_model.save(os.path.join(result_dir, 'vgg16_Final.h5'))



