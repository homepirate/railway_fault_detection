import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt


train_dir = 'archive/Railway Track fault Detection Updated/Train'
validation_dir = 'archive/Railway Track fault Detection Updated/Validation'
test_dir = 'archive/Railway Track fault Detection Updated/Test'

batch_size = 8
target_size = 224, 224



train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    # rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)


validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False


base_model_output = base_model.output
x = BatchNormalization()(base_model_output)
flat_layer = Flatten()(x)
# x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flat_layer)
x = Dense(512, activation='relu')(flat_layer)
x = Dropout(0.5)(x)
output = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size, epochs=50,
                    validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size)


train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Train accuracy: {train_accuracy}')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

model.save('my_model.keras')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xticks(np.arange(0, 50, 5))
# plt.yticks(np.arange(0, 1, 0.1))
plt.savefig('val_loss_plot.png')
plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xticks(np.arange(0, 50, 5))
plt.savefig('accuracy_plot.png')
plt.clf()



