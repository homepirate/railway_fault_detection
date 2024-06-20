import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import os
model = tf.keras.models.load_model('my_model.keras')


def predict_image(image_path):

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # добавляем измерение пакета

    img_array = img_array / 255.0  # нормализуем изображение

    prediction = model.predict(img_array)
    print(prediction)
    predicted_class = np.argmax(prediction)
    return 'Defective' if predicted_class == 0 else 'Non defective'

    # return predicted_class


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    folder_path = 'test/def'
    # folder_path = "archive/Railway Track fault Detection Updated/Test/Defective"
    files = os.listdir(folder_path)
    len_1 = len(files)
    c_1 = 0
    for file in files:
        if file.endswith(".jpg"):
            file_path = os.path.join(folder_path, file)
            if "Defective" == predict_image(file_path):
                c_1 += 1
            # print("Defected", predict_image(file_path))

    print("-" * 20)
    folder_path = 'test/notdef'
    # folder_path = "archive/Railway Track fault Detection Updated/Test/Non defective"
    files = os.listdir(folder_path)
    len_2 = len(files)
    c_2 = 0
    for file in files:
        if file.endswith(".jpg"):
            file_path = os.path.join(folder_path, file)
            if "Non defective" == predict_image(file_path):
                c_2 += 1
            # print("Non defected", predict_image(file_path))

    # print(f"Общий объем тестовых изображений: {len_1 + len_2}\nПроцент правильно предсказанных изображений: {(c_1 + c_2) / (len_1 + len_2)}")
    print(f"Процент правильно предсказанных изображений: {(c_1 + c_2) / (len_1 + len_2)}")


if __name__ == '__main__':
    main()
