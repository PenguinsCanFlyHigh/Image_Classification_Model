import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 데이터셋 경로 설정
data_dir = 'C:/Users/ubnzz/Downloads/recycle_dataset'

# 이미지 데이터 로드 및 전처리
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 데이터의 20%를 검증에 사용
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    # 클래스 이름을 명시적으로 지정하지 않아도 폴더 이름으로 자동 인식됩니다
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 클래스 이름 출력 (확인용)
class_names = train_ds.class_names
print(class_names)

# 성능을 위한 데이터셋 설정
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 모델 구성
num_classes = len(class_names)

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # 이미지 정규화
    layers.Conv2D(16, 3, padding='same', activation='relu'),  # 컨볼루션 레이어
    layers.MaxPooling2D(),  # 풀링 레이어
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),  # 일렬화
    layers.Dense(128, activation='relu'),  # 완전 연결 레이어
    layers.Dense(num_classes)  # 출력 레이어
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 구조 출력
model.summary()

# 모델 학습
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 결과 시각화 (선택 사항)
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='훈련 정확도')
plt.plot(epochs_range, val_acc, label='검증 정확도')
plt.legend(loc='lower right')
plt.title('훈련 및 검증 정확도')
plt.show()

model.save('recycle_image_classification_v1.h5')