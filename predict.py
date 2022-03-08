import cv2, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class Prediction:
    def __init__(self, full_data, model):
        self.full_data = full_data
        self.model = model

    def rle2mask(self, rle_string, img_shape=(256, 1600)):
        rle_array = np.array([int(s)for s in rle_string.split()])
        starts_array = rle_array[::2]-1
        lengths_array = rle_array[1::2]
        mask_array = np.zeros(img_shape[0]*img_shape[1],dtype=np.uint8)
        for i in range(len(starts_array)):
            mask_array[starts_array[i]:starts_array[i]+lengths_array[i]] = 1
        
        return mask_array.reshape(img_shape, order='F')

    def accuracy(self, y_true, y_pred):
        sum1 = 2 * np.sum(y_true * y_pred)
        sum2 = np.sum(y_true**2 + y_pred**2)
        dice = sum1 / sum2
        if np.isnan(dice):
            return 1.0
        else:
            return dice

    def visualize_model_prediction(self, path, img_id, show=False, save=False):
        fig, axs = plt.subplots(4, 3, figsize=(16, 8))
        img_obj = cv2.imread('./data/train_images/'+ img_id)
        masks_actual = self.full_data[self.full_data['ImageId'] == img_id]
        x = np.empty((1, 256, 1600, 3), dtype=np.float32) # image place-holders
        x[0,] = Image.open('./data/train_images/' + img_id)
        masks_predicted = self.model.predict(x)

        loss = []
        for i in range(4):
            y_true = self.rle2mask(masks_actual['rle_'+str(i+1)].iloc[0])
            y_pred = masks_predicted[0, :, :, i]
            y_pred[y_pred < 0.3] = 0
            
            sum1 = 2 * np.sum(y_true * y_pred)
            sum2 = np.sum(y_true**2 + y_pred**2)
            dice = sum1 / sum2
            if np.isnan(dice):
                loss.append(1.0)
            else:
                loss.append(dice)

        img_acc = sum(loss)/len(loss)

        for i in range(4):
            axs[i,0].imshow(img_obj)
            axs[i,0].set_title(img_id)
            axs[i,0].xaxis.set_visible(False)
            axs[i,0].yaxis.set_visible(False)
            axs[i,1].imshow(self.rle2mask(masks_actual['rle_'+str(i+1)].iloc[0]))
            axs[i,1].set_title("Actual mask for Class '{}'".format(i+1) )
            axs[i,1].xaxis.set_visible(False)
            axs[i,1].yaxis.set_visible(False)
            axs[i,2].imshow(masks_predicted[0,:,:,i])
            #axs[i,2].set_title("Predicted mask for Class '{}'".format(i+1))
            m1 = self.rle2mask(masks_actual['rle_'+str(i+1)].iloc[0])
            m2 = masks_predicted[0, :, :, i]
            axs[i,2].set_title("Predicted Class '{}', Acc:{:.3f}".format(i+1, self.accuracy(m1, m2)))
            axs[i,2].xaxis.set_visible(False)
            axs[i,2].yaxis.set_visible(False)

        if show:
            plt.show()
        if save:
            if not os.path.isdir('{}/pred_images'.format(path)):
                os.mkdir('{}/pred_images'.format(path))
            plt.savefig('{}/pred_images/pred_{}'.format(path, img_id))
        return img_acc, masks_predicted