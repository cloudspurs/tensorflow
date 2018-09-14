import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()


# normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()


epoch = list( i * 5 for i in range(1, 41))
one_acc = []
average_acc = []

for i in epoch:
	ten_acc = []
	for j in range(10):
		print('Number of times: ' + str(j))
		# configuring the layers of the model
		model = keras.Sequential([
		    keras.layers.Flatten(input_shape=(28, 28)),
		    keras.layers.Dense(128, activation=tf.nn.relu),
		    keras.layers.Dense(10, activation=tf.nn.softmax)
		])
		
		# compiling the model
		model.compile(optimizer=tf.train.AdamOptimizer(),  # Adaptive Moment Estimation
		              loss='sparse_categorical_crossentropy', # 稀疏多类对数损失
		              metrics=['accuracy'])

		model.fit(train_images, train_labels, epochs=i, verbose=2)
		test_loss, test_acc = model.evaluate(test_images, test_labels)
		print('Accuracy: ' + str(test_acc))
		ten_acc.append(test_acc)
	average_acc.append(np.mean(ten_acc))
	one_acc.append(ten_acc[0])
	
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epoch, one_acc, label='Once')
plt.plot(epoch, average_acc, label='Average 10')
plt.legend()
plt.savefig('acc.png')
#plt.show()

#print('\nTest accuracy:', test_acc)
#
#predictions = model.predict(test_images)
#print(predictions[0])
#print(np.argmax(predictions[0]))
#print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions,  test_labels)
#plt.show()
#
#i = 12
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions,  test_labels)
#plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
#num_rows = 5
#num_cols = 3
#num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#  plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  plot_image(i, predictions, test_labels, test_images)
#  plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  plot_value_array(i, predictions, test_labels)
#plt.show()


#img = test_images[0]
#print(img.shape)
#
#img = (np.expand_dims(img, 0))
#print(img.shape)
#
#predictions_single = model.predict(img)
#
#print(predictions_single)
#
#plot_value_array(0, predictions_single, test_labels)
#_ = plt.xticks(range(10), class_names, rotation=45)
#plt.show()
#
#print(np.argmax(predictions_single[0]))

