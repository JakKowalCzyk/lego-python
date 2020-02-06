from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from cnn import CNN
from sklearn.datasets import load_files

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset of faces")
ap.add_argument("-m", "--model", required = True, help = "path to output model")
ap.add_argument("-p", "--plot", required = True, help = "path to accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []
#
# def convert_image_to_array(files):
#     images_as_array=[]
#     for file in files:
#         # Convert to Numpy Array
#         images_as_array.append(img_to_array(load_img(file)))
#     return images_as_array
#
# data = load_files(args["dataset"])
# files = np.array(data['filenames'])
# train = np.array(convert_image_to_array(files))
# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = imutils.resize(image, width=64, height=64)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the label list
    label = imagePath.split(os.path.sep)[-2]
    label = "Legos" if label == "Legos" else "Bricks"
    labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis = 0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of the data
# for training and remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size = 0.20, stratify = labels, random_state = 42)

# initialize the model
print("[INFO] compiling model...")
model = CNN.build()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# train the network
print("[INFO] training network...")
checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
H = model.fit(trainX,trainY,
        batch_size = 32,
        epochs=30,
        validation_data=(testX, testY),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1),
    predictions.argmax(axis = 1), target_names = le.classes_))

# save the model to disk
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label = "acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
plt.show()