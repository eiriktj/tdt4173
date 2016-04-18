import sklearn
from sklearn import datasets, svm, tree, metrics, preprocessing
from scipy import ndimage
import numpy as np
import os
from PIL import Image
from decimal import Decimal

# 'svc' or 'dt'
classifier_type = 'svc'

def load_data():
    rootdir = os.getcwd()+'/chars74k-lite'
    test_images = []
    training_images = []

    for i in range(26):
        test_images.append([])
        training_images.append([])

    i = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print(os.path.join(subdir, file))
            label = ord(subdir[-1:]) - 97
            im = Image.open(os.path.join(subdir, file))
            data = list(im.getdata())
            if i%2:
                test_images[label].append([float(x) for x in data])
            else:
                training_images[label].append([float(x) for x in data])
            i += 1
    return test_images, training_images

def preprocess_data(data, prep_type='noise'):
    output = []
    if prep_type=='noise':
        for labels in data:
            tmp = []
            for image in labels:
                tmp.append(ndimage.median_filter(image, 3))
            output.append(tmp)
    elif prep_type=='scale':
        for labels in data:
            tmp = []
            for image in labels:
                tmp.append(preprocessing.scale(image))
            output.append(tmp)
    return output

# Load data
test, training =load_data()
# Convert to numpy arrays
test = np.array(test)
training = np.array(training)

# Preprocess data
preprocessing_type = 'noise'
test = preprocess_data(test, preprocessing_type)
training = preprocess_data(training, preprocessing_type)
preprocessing_type = 'scale'
test = preprocess_data(test, preprocessing_type)
training = preprocess_data(training, preprocessing_type)

training_target = []
x_training = []
test_target = []
x_test = []

for label in range(len(training)):
    for image in training[label]:
        training_target.append(label)
        x_training.append(image)

for label in range(len(test)):
    for image in test[label]:
        test_target.append(label)
        x_test.append(image)


if classifier_type=='svc':
    classifier = svm.SVC(gamma=0.005) #gamma = 0.004
elif classifier_type=='dt':
    classifier = tree.DecisionTreeClassifier(max_depth=20)

classifier.fit(x_training, training_target)
predicted = classifier.predict(x_test)

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_target, predicted)))
