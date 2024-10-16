import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 및 cuDNN을 사용할 수 있습니다.")
else:
    print("GPU를 찾을 수 없거나 cuDNN 설정이 올바르지 않습니다.")
