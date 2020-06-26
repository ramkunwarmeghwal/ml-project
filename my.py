from keras.datasets import mnist

dataset = mnist.load_data('mymnist.db')

train , test = dataset

X_train , y_train = train

X_test , y_test = test

import cv2

import matplotlib.pyplot as plt

X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')

from keras.utils.np_utils import to_categorical

y_train_cat = to_categorical(y_train)

y_train_cat

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(units=512, input_dim=28*28, activation='relu'))

model.summary()

def layers():

 x=0
 y=255
    
 model.add(Dense(random.randint(x, y), activation='relu'))

 model.add(Dense(units=random.randint(x, y), activation='relu'))

 model.add(Dense(units=random.randint(x, y), activation='relu'))



layers()

model.summary()

model.add(Dense(units=10, activation='softmax'))

model.summary()

from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )

h = model.fit(X_train, y_train_cat, epochs=1)


scores = model.evaluate(X_train, y_train_cat, verbose=1)
print('Test loss:', scores[0])
print('accuracy:', scores[1])


if scores[1] <= 0.80:
    layers()
else:
    pass


