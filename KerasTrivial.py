#coding: latin-1

# http://matplotlib.org/faq/virtualenv_faq.html
# Run me with frameworkpython

# For a single-input model with 2 classes (binary classification):
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np

size = 100

white = np.random.random((size, 2)) + [-50,-50]
blue = np.random.random((size, 2)) + [50,50]
data = np.concatenate( (white,blue))
labels = np.concatenate( (np.zeros(size),(np.zeros(size)+1) )  )


# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(data, labels, epochs=10000, batch_size=32)

print 'Keras Accuracy: %f' % (model.evaluate(data,labels)[1])
