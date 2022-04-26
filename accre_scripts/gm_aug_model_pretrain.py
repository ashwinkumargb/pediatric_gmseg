'''
Author: Ashwin Kumar (ashwin.kumar@vanderbilt.edu)
Description: Run GM model on augmented data based on early stopping with pretraining.
The model will save the results in a directory, specifically a plot of training and validation DSC,
DSC mean and std, history, and the model itself trained for that epoch amount.
'''
# %%
from keras.models import load_model
from model import *
from deepseg_gm import *
import numpy as np
import tensorflow as tf
from keras import Model
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
# %%
'''
Normalize volume to mean ~= 0 and std ~= 1
@input: volume to be normalized
@return: normalized volume
'''
def vol_norm(volume):
    """This method will enable the function call for the
        class object.

        :param volume: the volume to be normalized.
        """
    volume_mean = volume.mean()
    volume_std = volume.std()

    volume -= volume_mean
    volume /= volume_std

    return volume

# %%
'''
Reshape the matrix and fix the dimensionality
@input 3D array
@output: concatenated array
'''
def shape_and_norm(arr, norm = True):
    arr_sampled = []
    for i in range(arr.shape[2]):
        if norm:
            arr_sampled.append(vol_norm(np.reshape(arr[:,:,i], (1, 256, 256))))
        else:
            # Just resample no norm 
            arr_sampled.append(np.reshape(arr[:,:,i], (1, 256, 256)))
    return np.concatenate(arr_sampled, axis=0)

# %%
'''
Reshape a single slice to (1, 256, 256) and vice versa
@input: 2D array
@return: reshaped array
'''
def reshape_single_slice(arr, inverse = False):
    if not inverse:
        return np.reshape(arr, (1, 256, 256))
    else:
         return np.reshape(arr, (256, 256))

'''
Expand dimensions of last axis in numpy array
@input: arr
@output: return expanded numpy array
'''
def expand_dims(arr):
    return np.expand_dims(arr, axis=-1)

'''
Create directory and don't allow for repeats
@base_dir: base directorty string
@dir_name: directory name string
'''
def create_directory(base_dir, dir_name):
    if not os.path.isdir(os.path.join(base_dir, dir_name)):
    	os.mkdir(os.path.join(base_dir, dir_name))


# %%
bmodel = create_model(32)
opt = Adam(lr=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', min_delta=0.01,mode="max", patience=5)
bmodel.compile(optimizer=opt,
                loss=dice_coef_loss,
                metrics=['accuracy', dice_coef])
bmodel.load_weights('challenge_model.hdf5')

# %%
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
#bmodel.summary()
# %% [markdown]
# ### Load in the data and do some testing

# %%
#Load in the data
train_data = expand_dims(np.load('../data/train_data_aug.npy'))
train_targets = expand_dims(np.load('../data/train_targets_aug.npy').astype('float32'))
val_data = expand_dims(np.load('../data/val_data_aug.npy'))
val_targets = expand_dims(np.load('../data/val_targets_aug.npy').astype('float32'))
test_data = expand_dims(shape_and_norm(np.load('../data/test_data.npy')))
test_targets = expand_dims(shape_and_norm(np.load('../data/test_targets.npy'), False))

#Fit the model with epochs
history = Model.fit(bmodel, train_data, train_targets, batch_size=BATCH_SIZE, epochs=100, validation_data=(val_data, val_targets), callbacks=[callback])
epoch = int(len(history.history['val_dice_coef']))

#Save the model accordingly
save_file_name = '{}_epoch_aug'.format(epoch)
create_directory(os.getcwd(), save_file_name)
np.save('{}/history_{}.npy'.format(save_file_name, save_file_name), history.history)

# %%
(loss, accuracy, dsc) = Model.evaluate(bmodel, test_data, test_targets, verbose=1)

# %%
plt.plot(history.history['dice_coef'], label='dice_coef')
plt.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
plt.xlabel('Epoch')
plt.ylabel('dice_coef')
#plt.ylim([0.82, 0.89])
plt.legend(loc='lower right')
plt.savefig('{}/{}.png'.format(save_file_name, save_file_name))

# %%
#Calculate DSC
all_preds = Model.predict(bmodel, test_data)

# Reshape and threshold predictions
all_preds = threshold_predictions(np.reshape(all_preds, (196, 256, 256)))

def dice_coef_model_pred(y_true, y_pred):
    dice_smooth_factor = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + dice_smooth_factor) / (np.sum(y_true_f) + np.sum(y_pred_f) + dice_smooth_factor)


dsc_scores = []
for i in range(all_preds.shape[0]):
    # print(inputs)
    # print(dice_coef_model_pred(test_targets[i,:,:].astype('float64'),all_preds[i, :, :].astype('float64')))
    dsc_scores.append(dice_coef_model_pred(test_targets[i,:,:].astype('float64'), all_preds[i, :, :].astype('float64')))

#print(dsc_scores)
print('{:.4f}+-{:.4f}'.format(np.mean(dsc_scores),np.std(dsc_scores)))
np.save('{}/dsc_{}.npy'.format(save_file_name, save_file_name), np.array([np.mean(dsc_scores), np.std(dsc_scores)]))

# %%
# Save model
#os.chdir(os.getcwd() + '/' + save_file_name)
bmodel.save_weights('{}/trained_{}.h5'.format(save_file_name,save_file_name))

# %%



