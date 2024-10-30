import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model('image_classification_model_v1.h5')

# TFLite 변환기 설정
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 변환된 TFLite 모델 저장
with open('image_classification_model_v1.tflite', 'wb') as f:
    f.write(tflite_model)
