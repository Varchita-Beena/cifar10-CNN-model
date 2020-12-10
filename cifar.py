from keras import backend as K
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers

np.random.seed(2017)

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")


from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))

class_names = ['airplane','automobile','bird','cat','deer',
				'dog','frog','horse','ship','truck']
				

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
	ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
	idx = np.where(train_labels[:]==i)[0]
	features_idx = train_features[idx,::]
	img_num = np.random.randint(features_idx.shape[0])
	im = np.transpose(features_idx[img_num,::], (1, 2, 0))
	ax.set_title(class_names[i])
	plt.imshow(im)
plt.show()

train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255

# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)
K.clear_session()


def plot_model_history(model_history):
	fig, axs = plt.subplots(1,2,figsize=(15,5))
	# summarize history for accuracy
	axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
	axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
	axs[0].legend(['train', 'val'], loc='best')
	# summarize history for loss
	axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
	axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
	axs[1].legend(['train', 'val'], loc='best')
	plt.show()


def accuracy(test_x, test_y, model):
	result = model.predict(test_x)
	predicted_class = np.argmax(result, axis=1)
	true_class = np.argmax(test_y, axis=1)
	num_correct = np.sum(predicted_class == true_class) 
	accuracy = float(num_correct)/result.shape[0]
	return (accuracy * 100)

	
# Define the model
model = Sequential()

model.add(Conv2D(48, kernel_size = (3, 3), activation = 'relu', input_shape=(32,32,3), data_format = 'channels_last'))
#model.add(Conv2D(48, kernel_size = (3, 3), activation = 'tanh', input_shape=(32,32,3), data_format = 'channels_last'))
#model.add(Conv2D(48, kernel_size = (3, 3), activation = 'sigmoid', input_shape=(32,32,3), data_format = 'channels_last'))
#model.add(BatchNormalization())

#model.add(Conv2D(48, (3, 3),activation = 'relu'))
#model.add(Conv2D(48, (3, 3),activation = 'tanh'))
#model.add(Conv2D(48, (3, 3),activation = 'sigmoid'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), border_mode="same"))
model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Activation('tanh'))
#model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(96, (3, 3)))
#model.add(Activation('relu'))
#model.add(Activation('tanh'))
#model.add(Activation('sigmiod'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(192, (3, 3), border_mode="same"))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Activation('tanh'))
#model.add(Activation('sigmiod'))

#model.add(Conv2D(192, (3, 3)))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Activation('tanh'))
#model.add(Activation('sigmiod'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Activation('tanh'))
#model.add(Activation('sigmiod'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Activation('tanh'))
#model.add(Activation('sigmiod'))

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
#optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels, 
                       batch_size=128, epochs=5, 
                       validation_data = (test_features, test_labels))
end = time.time()


# plot model history
plot_model_history(model_info)
print ("Model took %0.2f seconds to train"%(end - start))
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))


