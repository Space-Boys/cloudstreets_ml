import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import sys

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


mypath = "/mnt/c/Work/NASA Challenge 2020/gibs_downloader/GibsDownloader/GibsDownloader/bin/Debug/out/2020-10-02/TileMatrix7"


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)
for file in onlyfiles:
    try:
        # Replace this with the path to your image
        image = Image.open(mypath+"/"+file)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        print(file+"\t"+str(prediction[0][0]))
    except:
        print(file+"\t0")