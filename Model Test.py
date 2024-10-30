import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. 모델 로드
loaded_model = tf.keras.models.load_model("image_classification_model_v1.h5")

# 2. 테스트 데이터 준비
test_dir = "C:/programming/Image_Classification_Model/images2"

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),  # 모델의 입력 크기
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3. 모델 평가
test_loss, test_accuracy = loaded_model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# 4. 개별 이미지 테스트 (선택사항)
img_path = "C:/programming/Image_Classification_Model/test_images/class_1/img1.jpg"

# 이미지 파일이 존재하는지 확인
if os.path.exists(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # 클래스 라벨 확인
    class_labels = list(test_generator.class_indices.keys())
    print(f"Predicted Class: {class_labels[predicted_class]}")
else:
    print(f"File not found: {img_path}")