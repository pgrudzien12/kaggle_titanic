import csv
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


with open('train.csv', newline='') as csvfile:
    train = list(csv.reader(csvfile))
with open('test.csv', newline='') as csvfile:
    test = list(csv.reader(csvfile))

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=[]):
        if(logs['accuracy']>0.85):
            print("I'm done here. Going home")
            self.model.stop_training = True

callbacks = MyCallbacks()
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = train[1:]
training_labels = train[0]

test_images = test[1:]
test_labels = test[0]

training_images = training_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([keras.layers.Flatten(), 
                                    keras.layers.Dense(128, activation=tf.nn.relu), 
                                    keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images) 
print(test_labels[0]) 

print(classifications[0])



print(train)
