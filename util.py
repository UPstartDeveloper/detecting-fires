# Imports
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd

def data_gen(df_gen, batch_size):
    """Generate batches of the dataset to train the model on, one subsection at a time.
       Credit goes to Milad Toutounchian for this implementation, originally found at:
       https://github.com/Make-School-Courses/DS-2.2-Deep-Learning/blob/master/Final_Project/image_data_prep.ipynb
    
       Parameters:
       df(DataFrame): larger portion of the datset used for training
       batch_size(int): the number of samples to include in each batch
       
       Returns:
       tuple: input features of the batch, along with corresponding labels
    
    """
    while True:
        # list of images
        x_batch = np.zeros((batch_size, 1024, 1024, 3))
        # list of labels
        y_batch = np.zeros((batch_size, 1))
        # add samples until we reach batch size
        for j in range(len(df_gen) // batch_size):
            batch_index = 0
            for index in df_gen['Unnamed: 0']:
                if batch_index < batch_size:
                    # add image to the input
                    filepath = f"Fire-Detection-Image-Dataset/{df_gen['Folder'][index]}/{df_gen['filename'][index]}"
                    img = Image.open(filepath)
                    image_red = img.resize((1024, 1024))
                    x_batch[batch_index] = img_to_array(image_red)
                    # set label
                    y_batch[batch_index] = df_gen['label'][index]
                    # increment index in the batch
                    batch_index += 1
            yield (x_batch, y_batch)