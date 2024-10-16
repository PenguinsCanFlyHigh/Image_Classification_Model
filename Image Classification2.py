import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# 데이터셋 경로 설정
data_dir = "C:/programming/Imageset model/images2"

# 이미지 전처리 및 데이터 로더 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 80% 학습, 20% 검증 데이터로 나누기
)

train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),  # ResNet50의 입력 크기
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 전이 학습을 위한 ResNet50 모델 불러오기 (사전 학습된 가중치 사용)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 전이 학습을 위한 모델 설정
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # 클래스 수 만큼의 출력 뉴런
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
