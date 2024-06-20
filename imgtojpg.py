from PIL import Image
import os

# Путь к папке с изображениями
image_folder = "archive/Railway Track fault Detection Updated/Validation/Defective"
# Формат, в который нужно преобразовать изображения
target_format = "JPEG"

# Создаем новую папку для сохранения преобразованных изображений
output_folder = "archive/Railway Track fault Detection Updated/Validation/Defective"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Проходим по всем файлам в папке с изображениями
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        # Открываем изображение
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)

        # Проверяем режим цветности и конвертируем в RGB при необходимости
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Преобразуем изображение в целевой формат и сохраняем его
        new_filename = os.path.splitext(filename)[0] + "." + target_format.lower()
        output_path = os.path.join(output_folder, new_filename)
        img.save(output_path, format=target_format)
        print(f"Файл {filename} успешно сконвертирован в формат {target_format}")

print("Конвертация завершена!")