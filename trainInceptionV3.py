import argparse
import os

from keras import Input
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lenet import LeNet

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", required=True, help="path to accuracy/loss plot")
args = vars(ap.parse_args())

data = []
labels = []
#

for imagePath in sorted(list(paths.list_images(args["dataset"]))):

    image = load_img(imagePath, target_size=(128,128))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = "Legos" if label == "Legos" else "Bricks"
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

print("[INFO] compiling model...")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

new_input = Input(shape=(128, 128, 3))
model = InceptionV3(weights=None, input_tensor=new_input, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight, batch_size=34, epochs=40, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
plt.show()
