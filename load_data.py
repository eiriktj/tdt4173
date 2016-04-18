import sklearn
import os
from PIL import Image
from decimal import Decimal

def load_data():
    rootdir = os.getcwd()+'/chars74k-lite'
    test_images = [[]]*26
    training_images = [[]]*26

    i = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print(os.path.join(subdir, file))
            label = ord(subdir[-1:]) - 97
            im = Image.open(os.path.join(subdir, file))
            data = list(im.getdata())
            if i%2:
                test_images[label].append(data)
            else:
                training_images[label].append(data)
            i += 1
    return test_images, training_images

def process_data(image_list):
    for x in range(len(image_list)):
        for y in range(len(image_list[x])):
            for i in range(len(image_list[x][y])):
                image_list[x][y][i] = str(float(image_list[x][y][i])/255)
                
test, training =load_data()
process_data(test)

print test[0][0]



