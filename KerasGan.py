#coding: latin-1

# http://matplotlib.org/faq/virtualenv_faq.html
# Run me with frameworkpython

# For a single-input model with 2 classes (binary classification):
from keras.models import Sequential
from keras.layers import Dense


def generator(img_shape, z_dim):
  model = Sequential()
  # Hidden layer
  model.add(Dense(128, input_dim = z_dim))
  # Leaky ReLU
  model.add(LeakyReLU(alpha=0.01))
  # Output layer with tanh activation
  model.add(Dense(28*28*1, activation='tanh'))
  model.add(Reshape(img_shape))
  z = Input(shape=(z_dim,))
  img = model(z)
  return Model(z, img)