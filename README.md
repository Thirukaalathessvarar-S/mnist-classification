# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values.

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test .Build the convolutional neural network model

### STEP 6:
Train the model with the training data.Plot the performance plot

### STEP 7:
Evaluate the model with the testing data.Fit the model and predict the single input

## PROGRAM
```
Developed by : Thirukaalathessvarar S
Red.no : 212222230161
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape

X_test.shape

single_image= X_train[0]
     
single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()

# Add a convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add a max-pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Add another convolutional layer with 64 filters and a 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add another max-pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to feed it into a dense layer
model.add(layers.Flatten())

# Add a dense layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Add an output layer with 10 units for the 10 classes and softmax activation
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
     
model.fit(X_train_scaled ,y_train_onehot, epochs=8,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

# Prediction for a single input
img = image.load_img('three.png')
type(img)

img = image.load_img('three.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![dl1_exp1](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/aae01f4e-156b-4b3a-afdd-4ff31ac3eeea)

![dl2_exp3](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/572e228b-ede5-42d3-9032-cea43b1a31fa)


### Classification Report
![dl3_exp3](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/b76a9966-aeba-40cc-9868-c14ffb36e221)



### Confusion Matrix
![dl4_exp3](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/d6a1aa7e-d67d-4606-a2c8-465903319b9a)


### New Sample Data Prediction
.![dl5_exp3](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/7f9d94bf-0206-49ba-aca8-933ce7a6bf70)

![dl6_exp3](https://github.com/Thirukaalathessvarar-S/mnist-classification/assets/121166390/fd1bcd8f-4ea6-43ff-b93e-9d7129ff9eee)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
