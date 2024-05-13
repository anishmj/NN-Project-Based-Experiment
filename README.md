#Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.


## PROGRAM
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
     

single_image= X_train[1520]
     

single_image.shape
     

plt.imshow(single_image,cmap='gray')
     

y_train.shape
     

X_train.min()
     

X_train.max()
     

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
     

X_train_scaled.min()
     

X_train_scaled.max()
     

y_train[10]
     

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
     

type(y_train_onehot)
     

y_train_onehot.shape
     

single_image = X_train[1560]
plt.imshow(single_image,cmap='gray')
     

y_train_onehot[10]
     

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
     
model = keras.Sequential()
input =keras.Input(shape=(28,28,1))
model.add(input)
layer1 = layers.Conv2D(filters =32 , kernel_size =(5,5),strides =(1,1),padding ='same')
model.add(layer1)
pool1 = layers.MaxPool2D(pool_size=(2,2))
model.add(pool1)
layer2 = layers.Conv2D(filters =16 , kernel_size =(5,5),strides =(1,1),padding ='same')
model.add(layer2)
layer3 = layers.Flatten()
model.add(layer3)
hidden1 =layers.Dense(units =8, activation='relu')
model.add(hidden1)
output = layers.Dense(units=10,activation='softmax')
model.add(output)
     

model.summary()
     
model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=15,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
     

metrics = pd.DataFrame(model.history.history)
     

metrics.head()
     

metrics[['accuracy','val_accuracy']].plot()
     

metrics[['loss','val_loss']].plot()
     

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
     

print(confusion_matrix(y_test,x_test_predictions))
     

print(classification_report(y_test,x_test_predictions))
     
# Prediction for a single input


img = image.load_img('/content/seven1.jpg')     

type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

     

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
     

print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

```

##  Inverting the image

```
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

     
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![OP](OP1.png)
### Classification Report

![OP](OP2.png)

### Confusion Matrix

![OP](OP3.png)

### New Sample Data Prediction

![OP](OP4.png)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.



