import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def rle2mask(rle_string, img_shape=(256, 1600)):
    rle_array = np.array([int(s)for s in rle_string.split()])
    starts_array = rle_array[::2]-1
    lengths_array = rle_array[1::2]
    mask_array = np.zeros(img_shape[0]*img_shape[1],dtype=np.uint8)
    for i in range(len(starts_array)):
        mask_array[starts_array[i]:starts_array[i]+lengths_array[i]] = 1
    
    return mask_array.reshape(img_shape, order='F')

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=4, num_classes=None, shuffle=True, preprocess=None):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        self.preprocess = preprocess
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // (self.batch_size)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        train_datagen = ImageDataGenerator()
        param = {'flip_horizontal':True, 'samplewise_std_normalization' : True, 'width_shift_range':0.1, 'height_shift_range':0.1}

        X = np.empty((self.batch_size, 256, 1600, 3), dtype=np.float32) # image place-holders
        Y = np.empty((self.batch_size, 256, 1600, 4), dtype=np.float32) # 4 masks place-holders

        for i, id in enumerate(batch):
            img = Image.open('./data/train_images/' + str(self.df['ImageId'].loc[id]))
            X[i,] = train_datagen.apply_transform(x=img, transform_parameters=param) #input image
            for j in range(4):
                mask = rle2mask(self.df['rle_'+str(j+1)].loc[id])
                Y[i,:,:,j] = train_datagen.apply_transform(x=mask, transform_parameters=param) #mask for each class
 
        # preprocess input
        if self.preprocess!=None: X = self.preprocess(X)

        return X, Y