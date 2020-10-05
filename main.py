import math
import re
import csv

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from glob import glob
from PIL import ImageFile

from model_training import create_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
model = create_model()


class ImageSequence(tf.keras.utils.Sequence):

    def __init__(self, data_path, batch_size=64, dim=(256, 256)):
        self.img_list = []

        for filename in glob(f'{data_path}\\*.jpg'):
            self.img_list.append(filename)

        self.batch_size = batch_size
        self.dim = dim

    def __len__(self):
        """Denotes number of batches per img_list"""
        # Ceil to include remainders that do not fit batch_size
        return math.ceil(len(self.img_list) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        # batch_img of size bach_size
        batch_img = self.img_list[start_index: end_index]

        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        # Generate data
        for j, filename in enumerate(batch_img):
            # Store sample
            X[j, ] = img_to_array(
                load_img(filename, target_size=self.dim)
            )

        return X


def build_predictions(img_folder):
    prediction_ds = ImageSequence(img_folder)

    predictions = model.predict(prediction_ds)
    return tf.nn.softmax(predictions)


threshold = 0.5
date = '2020-10-03'
for tile_matrix_folder in Path(f'datasets\\classifications\\{date}').glob('TileMatrix[0-7]'):
    # Get level
    level_match = re.search(r'TileMatrix(\d)', tile_matrix_folder.name)
    if level_match:
        level = level_match.group(1)
        scores = build_predictions(tile_matrix_folder)

        with open(tile_matrix_folder.joinpath('hazards.tsv'), 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(['Date', 'Level', 'Row', 'Column', 'Probability'])

            for i, img in enumerate(tile_matrix_folder.glob("*.jpg")):
                score = scores[i][1].numpy()

                if score < threshold:
                    continue

                row_col_match = re.search(r'(\d+)_(\d+)', img.name)
                if row_col_match:
                    row = row_col_match.group(1)
                    col = row_col_match.group(2)

                    writer.writerow([date, level, row, col, score])
