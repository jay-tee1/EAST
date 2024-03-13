import os
import random
import shutil

#get all files of current directory
files = os.listdir()

#take only images, that have a corresponding json file and are thus labeld
images = [file for file in files if file.endswith('.jpeg') and f"{file.removesuffix('.jpeg')}.json" in files]
#shuffel for randomised distribution
random.shuffle(images)

#get the 70 percent of the labeld data
train_amount = round(len(images) * 0.7)

#split according to amount of images
train_images = images[:train_amount]
test_images = images[train_amount:]

assert len(train_images) == train_amount
assert len(train_images) + len(test_images) == len(images)

os.makedirs('train_images')
os.makedirs('test_images')

#copy images into folders
for train_image in train_images:
    shutil.copy(train_image, 'train_images')
    shutil.copy(f"{train_image.removesuffix('.jpeg')}.json", 'train_images')

for test_image in test_images:
    shutil.copy(test_image, 'test_images')
    shutil.copy(f"{test_image.removesuffix('.jpeg')}.json", 'test_images')

assert len(os.listdir('train_images')) == len(train_images) * 2
assert len(os.listdir('test_images')) == len(test_images) * 2