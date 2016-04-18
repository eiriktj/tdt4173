import sklearn

from PIL import Image

im = Image.open("/home/ole/Desktop/chars74k-lite/a/a_414.jpg")
im.show()

im2 = list(im.getdata())

print(im2)
