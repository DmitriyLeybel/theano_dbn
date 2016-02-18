from descent2 import x,y
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

x, y = x.reshape(x.shape[0],1), y.reshape(y.shape[0],1)
xT, yT = x[20:30], y[20:30]
x, y = x[0:20],y[0:20]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=1, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

model.fit(x, y,
          nb_epoch=20,
          batch_size=16,
          show_accuracy=True)
score = model.evaluate(xT, yT, batch_size=16)

print(model.predict(xT[1:10]))