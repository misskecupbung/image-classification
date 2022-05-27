
# Import library yang diperlukan
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import warnings
warnings.filterwarnings("ignore")

# Mount ke drive
#from google.colab import drive
#drive.mount('/content/drive')

# Set datadir tempat data train berada dan variabel
#DATADIR = '/content/drive/MyDrive/Colab Notebooks/trainingSample/'
DATADIR = '/home/dwi/Documents/datmin/trainingSample'

IMAGE_SHAPE = (224, 224)

# Rescale image dan split menjadi data train dan validation
datagen_kwargs = dict(rescale=1./255, validation_split=.20)

# Buat train_generator dan valid_generator
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    DATADIR, 
    subset="validation", 
    shuffle=True,
   target_size=IMAGE_SHAPE
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    DATADIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE
    )

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

# Tampilkan train_generator 
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

# Buat model klasifikasi
# Disini menggunakan TensorFlow hub untuk me-load pre-trained model
model = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                 output_shape=[1280],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
model.build([None, 224, 224, 3])

model.summary()

# Compile model
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=['acc'])

# Train model
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

hist = model.fit(
    train_generator, 
    epochs=100,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch).history

# Cek akurasi model
final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
print("Final loss: {:.2f}".format(final_loss))
print("Final akurasi: {:.2f}%".format(final_accuracy * 100))

# Plotting grafik untuk mengetahui seberapa bagus training dan validasi
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,50])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])

val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)
print("Validation batch shape:", val_image_batch.shape)

dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)

tf_model_predictions = model.predict(val_image_batch)
print("Prediction results shape:", tf_model_predictions.shape)

predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
print(predicted_labels)

# Cek performa model
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range((len(predicted_labels)-2)):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Prediksi model (hijau: benar, merah: salah)")

# Simpan model ke dalam file pickle
pickle.dump(tf_model_predictions, open(DATADIR+"/model.pkl", "wb"))
