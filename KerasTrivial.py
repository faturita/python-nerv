#coding: latin-1

# This is a basic keras sample but describes the basic of using Deep Learning for ML classification

from numpy.random import seed
#seed(1)


# For a single-input model with 2 classes (binary classification):
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Generate dummy data
import numpy as np

masscenterwhite = [-50,-50]
masscenterblue = [50,50]

size = 100

white = np.random.random((size, 2)) + masscenterwhite
blue = np.random.random((size, 2)) + masscenterblue
data = np.concatenate( (white,blue))
labels = np.concatenate( (np.zeros(size),(np.zeros(size)+1) )  )


# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(data, labels, epochs=320, batch_size=32,validation_split=0.4)

print 'Keras Accuracy: %f' % (model.evaluate(data,labels)[1])

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


size = 40
white = np.random.random((size, 2)) + masscenterwhite
blue = np.random.random((size, 2)) + masscenterblue
data = np.concatenate( (white,blue))
testlabels = np.concatenate( (np.zeros(size),(np.zeros(size)+1) )  )

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(white[:,0], white[:,1], s=10, c='b', marker="x", label='White')
ax1.scatter(blue[:,0], blue[:,1], s=10, c='r', marker="o", label='Blue')
plt.xlabel('White')
plt.ylabel('Blue')
plt.legend(loc='upper left');
plt.show()


from sklearn.metrics import confusion_matrix

predlabels = model.predict(data, batch_size=32)

print predlabels

predlabels = predlabels.round()

print predlabels

C = confusion_matrix(testlabels, predlabels)
acc = (float(C[0,0])+float(C[1,1])) / ( data.shape[0])
print 'Accuracy: %f' % (acc)
print(C)