import asyncio
import os
import sys

from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import logging


from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.types import FSInputFile


TOKEN = ''
model = tf.keras.models.load_model('my_model.keras')


dp = Dispatcher()


def predict_image(image_path):

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # добавляем измерение пакета

    img_array = img_array / 255.0  # нормализуем изображение

    prediction = model.predict(img_array)
    print(prediction)
    predicted_class = np.argmax(prediction)
    return 'Defective' if predicted_class == 0 else 'Non defective'


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!\nPlease send IMG")


@dp.message(F.content_type == ContentType.PHOTO)
async def get_photo_from_user(message: Message, bot: Bot):
    file_name = f"{message.photo[-1].file_id}.jpg"
    await message.bot.download(file=message.photo[-1].file_id, destination=file_name)

    result = predict_image(file_name)
    await bot.send_photo(chat_id=message.from_user.id, photo=FSInputFile(path=file_name), caption=result)

    os.remove(file_name)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())

