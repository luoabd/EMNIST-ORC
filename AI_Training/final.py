#import Tensorflow and keras
import tensorflow as tf

#import other libraries
import numpy as np
import matplotlib.pyplot as plt
from fg import freeze_graph
from mnist import MNIST

mndata = MNIST('data')

#Data Sets
EMNIST_TRAINING_IMAGES = "data/emnist-letters-train-images-idx3-ubyte"
EMNIST_TRAINING_LABELS = "data/emnist-letters-train-labels-idx1-ubyte"
EMNIST_TEST_IMAGES = "data/emnist-letters-test-images-idx3-ubyte"
EMNIST_TEST_LABELS = "data/emnist-letters-test-labels-idx1-ubyte"

#Load data
X_train, y_train = mndata.load(EMNIST_TRAINING_IMAGES, EMNIST_TRAINING_LABELS)
X_test, y_test = mndata.load(EMNIST_TEST_IMAGES, EMNIST_TEST_LABELS)

#Convert data to numpy arrays
#Also normalize the images so they become between [0, 1]
X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)

#Resize the images to 28*28 for pre-processing
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

#for train data
for t in range(112800):
    X_train[t]=np.transpose(X_train[t])

#for test data
for t in range(14800):
    X_test[t]=np.transpose(X_test[t])

print('Process Complete: Rotated and reversed test and train images!')

#test to see if labels are correct
characters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(characters[y_train[i]-1])
plt.show()

#Reshape again for the model
X_train = X_train.reshape(X_train.shape[0], 784, 1)
X_test = X_test.reshape(X_test.shape[0], 784, 1)

#Start building the model
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM
from keras import backend as K
from keras.constraints import maxnorm
def resh(ipar):
    opar = []
    for image in ipar:
        opar.append(image.reshape(-1))
    return np.asarray(opar)

from keras.utils import np_utils

train_images = X_train.astype('float32')
test_images = X_test.astype('float32')

train_images = resh(train_images)
test_images = resh(test_images)


train_labels = np_utils.to_categorical(y_train, 27)
test_labels = np_utils.to_categorical(y_test, 27)


K.set_learning_phase(1)

model = Sequential()

model.add(Reshape((28,28,1), input_shape=(784,)))
model.add(Convolution2D(32, (5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(27, activation='softmax'))

print('Model Created Succesfully')

history = model.fit(train_images,train_labels,validation_data=(test_images, test_labels), batch_size=512, epochs=20)

#evaluating model on test data. will take time
scores = model.evaluate(test_images,test_labels, verbose = 0)
print("Accuracy: %.2f%%"%(scores[1]*100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name])
tf.train.write_graph(frozen_graph,'.','../output.pb',as_text=False)
print(model.input.op.name)
print(model.output.op.name)
