import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from warnings import simplefilter

simplefilter(action='ignore')

# LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# train_images: numpy.ndarray

train_images.shape

# Let's look at one image
IMG_INDEX = 9

plt.figure(figsize=(2, 2))  # Just add figure figsize

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)

plt.xlabel(class_names[train_labels[IMG_INDEX][0]])

plt.show()

# Matplotlib: Image Subplot Loop

import matplotlib.pyplot as plt
import numpy as np

# Create an array of images
images = np.random.randint(100, size=10)

# Determine the number of rows and columns for the subplots
num_images = len(images)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))

# Create a figure and axis objects for the subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5, 5))

# Loop through the images and plot them on the appropriate subplot
for i in range(num_images):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].imshow(train_images[images[i]])
    axs[row, col].axis('off')

# Show the plot
plt.show()

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())  # Flatten!

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))

model.summary()

EPOCHS = 4

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

acc = test_acc * 100

print(f"\nTest Accuracy: {acc} %")

# from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# creates a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# pick an image to transform
test_img = train_images[20]
img = image.img_to_array(test_img)  # convert image to numpy arry
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

# this loop runs forever until we break, saving images to current directory with specified prefix
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# Split the data manually into 80% training, 10% testing, 10% validation

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Corrupt JPEG data: 162 extraneous bytes before marker 0xd9

# creates a function object that we can use to get labels
get_label_name = metadata.features['label'].int2str

# Display 5 images from the dataset
for image, label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(image, label):
    """
    Returns an image that is reshaped to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)

    image = (image / 127.5) - 1

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label

train = raw_train.map(format_example)

validation = raw_validation.map(format_example)

test = raw_test.map(format_example)

for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

validation_batches = validation.batch(BATCH_SIZE)

test_batches = test.batch(BATCH_SIZE)

for img, label in raw_train.take(2):
  print("\nOriginal shape:", img.shape)

for img, label in train.take(2):
  print("\nNew shape:", img.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()

image_batch = np.expand_dims(image, axis=0)

feature_batch = base_model(image_batch)

print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.summary()

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before
# training it on our new images

initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("\nLoss:", loss0)

print("\nAccuracy:", accuracy0)

# Last epoch: loss: 0.8533 - accuracy: 0.4781
# Now we've trained the model, and we get better values.

# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']

print("\nAccuracy:", acc)

model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()