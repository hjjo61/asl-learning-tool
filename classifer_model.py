import tensorflow as tf
import os # used to navigate file structures
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import normalize, to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Avoid OOM errors by setting GPU Memory Consumption Growth (out of memory)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_path = 'data'
image_exts = ['jpg','jpeg']

for image_class in os.listdir(data_path):
    for image in os.listdir(os.path.join(data_path, image_class)):
        image_path = os.path.join(data_path, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# load dataset from directory - keras builds it for you and does preprocessing
data = tf.keras.utils.image_dataset_from_directory('data', image_size = (300,300)) # data generator
# train_datagen = ImageDataGenerator(rescale=1.0/255, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
# train_generator = train_datagen.flow_from_directory(data_path, target_size=(300,300), batch_size=32, class_mode='categorical', shuffle=True)
# labels = {value: key for key, value in train_generator.class_indices.items()}


# print("Label mappings")
# for key, value in labels.items():
#     print(f"{key}:{value}")

# PREPROCESSING DATA
# note: when building deep learning models for image classification
# you want your rgb values to be as small as possible instead of 0 - 255
# this helps a ton with optimization

# data.map lets you perform a transformation
# when you go get your data, this is the temporary function you do to your data
data = data.map(lambda x, y: (x / 255, y))

# fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
# idx = 0

# for i in range(2):
#     for j in range(5):
#         label = labels[np.argmax(train_generator[0][1][idx])]
#         ax[i, j].set_title(f"{label}")
#         ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
#         ax[i, j].axis("off")
#         idx += 1

# plt.tight_layout()
# plt.suptitle("Sample Training Images", fontsize=21)
# plt.show()

# split data batches into train and test
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
print(train_size)
print(val_size)
print(test_size)

# TODO: look into shuffling data too !!!!
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)



# BUILDING THE MODEL
model = Sequential()

# 16 filters, 3x3 pixels in size, stride 1
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dense(4))

model.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()


# logdir = 'logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train,  epochs=20, validation_data=val)