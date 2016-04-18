import sklearn
import os
from PIL import Image
from decimal import Decimal

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

def preprocess_data(image_list):
    for label in image_list:
        for image in label:
            for p in range(len(image)):
                image[p] = image[p]/255.0

test, training =load_data()
preprocess_data(test)
preprocess_data(training)


print test[0][0][0]
print test[1][0][0]


