# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website:
# https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


def eval_performance(cm):
    """eval_performance (cm)

    Function that calculates the performance of a ML Model
    by getting the Accuracy, Precision, Recall and F1 Score values

    Arguments:
        cm {List} -- The Confusion Matrix
    """
    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"True Positives: {tp}\nTrue Negatives: {tn}\n" +
          f"False Positives: {fp}\nFalse Negatives: {fn}\n")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\n" +
          f"F1 Score: {f1_score}")

# Part 1 - Building the CNN


# Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                             input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a Second Convolutional Layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                             activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 = Fitting the CNN to the images
# Code from https://keras.io/preprocessing/image/

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

taining_set = train_datagen.flow_from_directory(
    '/home/baxtor95/ML_Course/Machine Learning A-Z/Part 8 - ' +
    'Deep Learning/Section 40 - Convolutional Neural Networks ' +
    '(CNN)/Convolutional_Neural_Networks/dataset/training_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory(
    '/home/baxtor95/ML_Course/Machine Learning A-Z/Part 8 - ' +
    'Deep Learning/Section 40 - Convolutional Neural Networks ' +
    '(CNN)/Convolutional_Neural_Networks/dataset/test_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(taining_set, steps_per_epoch=8000, epochs=25,
                         validation_data=test_set, validation_steps=2000)
