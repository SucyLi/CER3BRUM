# specify the location of the library
lib_path = '.'

import os

# garbage collection should prevent memory leaks during training exploiting generators
import gc

import sys
sys.path.insert(0, lib_path)

import numpy as np
import keras
import tensorflow as tf
import nibabel as nib

from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle

import random
import time
import argparse
import json

# useful for logging
from datetime import datetime

# import the CEREBRUM library

from cer3brum_lib.vol_losses import losses_dict
from cer3brum_lib import utils

src_path = '/media/ebs/CER3BRUM/src'
os.chdir(src_path+'/cer3brum_lib')
print(os.getcwd)
from fullyvol import cer3brum
os.chdir(src_path)

# tuple containing the dimension of each volume (tuple of three integers)
data_dims = (256, 320, 320)

# for sake of clarity in the following code, define:
num_slices_sag = data_dims[0]
num_slices_cor = data_dims[1]
num_slices_lon = data_dims[2]

# (int)
num_segmask_labels = 7

# (str)
dataset_name = 'hcp' #'hcp_raw'

# (bool)
use_standardisation = True

# (str)
anat_suffix = '_3T_T1w_MPR1_bc_n4.nii.gz' #'_3T_T1w_MPR1.nii.gz'

# (str)
segm_suffix = '_final_contr.nii.gz'

# (int)
num_epochs = 50

# (float)
learning_rate = 0.00042

# (int)
num_filters = 16

# (str - to choose from Keras activation functions: https://keras.io/activations/)
encoder_act_funct = 'relu'

# (str - to choose from Keras activation functions: https://keras.io/activations/)
decoder_act_funct = 'relu'

# (str - to choose between 'categorical_crossentropy', 'multiclass_dice_coeff',
# 'weighted_multiclass_dice_coeff', 'tversky_loss', 'tanimoto_coefficient').
loss_funct = 'categorical_crossentropy'

# (str - to choose between 'strconv' and 'maxpool')
model_arch = 'strconv'

# (int)
training_samples = 115

# (int)
val_samples = 15

# (str)
classification_act_funct = 'softmax'

# change this to match your dataset location
data_path = os.path.join('/media/ebs/data', dataset_name)

# specify the folders containing the training set and the validation set
training_path = os.path.join(data_path, 'training')
validation_path = os.path.join(data_path, 'validation')

# build a model dictionary (for sake of clarity of the user-defined parameters)
models_dict = {'maxpool' : cer3brum.ThreeLevelsMaxPool,
               'strconv' : cer3brum.ThreeLevelsStrConv,
              }
               
names_dict = {'maxpool' : 'c3rebrum_maxpool_%df_%d'%(num_filters, training_samples),
              'strconv' : 'c3rebrum_strconv_%df_%d'%(num_filters, training_samples),
              }

losses_names_dict = {'categorical_crossentropy'       : '_cc',
                     'multiclass_dice_coeff'          : '_sum_dc',
                     'average_multiclass_dice_coeff'  : '_avg_dc',
                     'weighted_multiclass_dice_coeff' : '_wght_dc',
                     'tversky_loss'                   : '_tsv',
                     'tanimoto_coefficient'           : '_tc'}

# the model name is function of the chosen hyperparameters
model_name = names_dict[model_arch] + losses_names_dict[loss_funct] + '_pp.h5'

models_dir  = os.path.join('../output/models/', dataset_name)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logs_dir = os.path.join('../output/logs', dataset_name)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
         
model_path    = os.path.join(models_dir, model_name)

# location of the "_LOG.json" file - where all the training param.s will be logged
logfile_path  = os.path.join(logs_dir, model_name[0:-3] + '_LOG.json')

# location of the "_ARCH.txt" file - where the output of Kerad model.summary() will be saved
archfile_path = os.path.join(logs_dir, model_name[0:-3] + '_ARCH.txt')

# training directory and validation directory structure (one anatomical + segmentation per subdir)
training_subdir_list = list() 
training_volumes_path = list()
training_subdir_list = sorted(os.listdir(training_path))
training_set_size = len(training_subdir_list)
for idx in training_subdir_list:
    training_volumes_path.append(os.path.join(training_path, idx))

validation_subdir_list = list() 
validation_volumes_path = list()
validation_subdir_list = sorted(os.listdir(validation_path))[0:val_samples]
validation_set_size = len(validation_subdir_list)
for idx in validation_subdir_list:
    validation_volumes_path.append(os.path.join(validation_path, idx))

print('Training on %d volumes, validating on %d (%s)'%(training_set_size, validation_set_size, dataset_name))



stdz_matrices_path = os.path.join('../output/zscoring/', dataset_name)

# in this case, mean and std are computed on the first 250 training volumes of the dataset
voxelwise_mean_path = os.path.join(stdz_matrices_path, 'voxelwise_mean.nii.gz')
voxelwise_std_path = os.path.join(stdz_matrices_path, 'voxelwise_std.nii.gz')

if not os.path.exists(voxelwise_mean_path) or not os.path.exists(voxelwise_std_path):
    print("ERROR: voxelwise mean and standard deviation volumes not found.")
    print('Have you run the data notebook?')    
    sys.exit(0)

# to make the data suitable for pooling, reduce the last dimension so that it can be divided by 2^3
voxelwise_mean = np.array(nib.load(voxelwise_mean_path).dataobj[:, :, :]).astype(dtype = 'float32')
voxelwise_std = np.array(nib.load(voxelwise_std_path).dataobj[:, :, :]).astype(dtype = 'float32')

# zero-padding is added by some pipelines at the borders, thus the voxelwise variability
# would result zero. Change these values to 1 in order to mantain the zero in the division
# (and not give rise to NaNs)
voxelwise_std[voxelwise_std == 0] = 1

# check volume integrity (presence of NaN and inf could arise during the standardisation)
def check_vol_integrity(input_vol, vol_name):

    if np.sum(np.isnan(input_vol)) != 0:
        print('WARNING: %d NaN(s) found in volume "%s"!'%(np.sum(np.isnan(input_vol)), vol_name))
        sys.exit(0)
        
    if np.sum(np.isinf(input_vol)) != 0:
        print('WARNING: %d inf(s) found in volume "%s"!'%(np.sum(np.isinf(input_vol)), vol_name))
        sys.exit(0)

## ----------------------------------------

# volumes z-scoring 
def volume_zscoring(input_vol, voxelwise_mean, voxelwise_std):

    # standardize each training volume
    input_vol -= voxelwise_mean
    input_vol /= voxelwise_std
    
    return input_vol

## ----------------------------------------

# create a generator that handles data loading dynamically in order to fit the entire dataset into RAM
def vol_generator(vol_list, data_dims, num_segmask_labels, batch_size,
                  use_standardisation, voxelwise_mean, voxelwise_std):
    
    # "batch_size" here is not the "actual" batch size (onto which the gradient is computed),
    # but just the number of volumes to load dinamically at once
    num_volumes = len(vol_list)
    
    anat_suffix = '_3T_T1w_MPR1_bc_n4.nii.gz' #'_3T_T1w_MPR1.nii.gz'

    segm_suffix = '_final_contr.nii.gz'
    
    while 1: 

        shuffle(vol_list)

        # if "num_samples" cannot be divided without remainder by "batch_size", some samples are discarded
        # (at mostr batch_size - 1); if batch_size = 1 (as in the fully volumetric case) then all the dataset
        # is loaded during the execution
        for offset in range(0, num_volumes, batch_size):
            
            # define the batch that will be loaded when the yield instruction is executed (vol paths)
            batch_volumes_list = vol_list[offset : offset+batch_size]

            # init the list that will contain the samples
            x_train_batch = np.zeros((batch_size, data_dims[0], data_dims[1], data_dims[2], 1), dtype=np.float)
            y_train_batch = np.zeros((batch_size, data_dims[0], data_dims[1], data_dims[2], num_segmask_labels), dtype=np.float)

            # for every sample in the selected chunk
            for vol_num, vol in enumerate(batch_volumes_list):
                
                vol_name = vol.split('/')[-1]

                mri_path = os.path.join(vol, vol_name + anat_suffix)
                segmask_path = os.path.join(vol, vol_name + segm_suffix)  

                # load the actual volume cropping to 168 in the last dimension to avoid problems during pooling
                temp = np.array(nib.load(mri_path).dataobj[:, :, :]).astype(dtype = 'float32')

                # check if everything is ok
                if temp.shape != data_dims:
                    print('\n Warning: volume "%s" size mismatch: skipping to the next volume...'%(vol_name))
                    continue
                
                if use_standardisation == True:
                    temp = volume_zscoring(temp, voxelwise_mean, voxelwise_std)

                check_vol_integrity(temp, vol_name)

                x_train_batch[vol_num, :, :, :, 0] = temp

                seg = np.array(nib.load(segmask_path).dataobj[:, :, :]).astype(dtype = 'uint8')
        
                # #  as we don't have many cases of white matter lesions (class 4) merge this class with white matter (class 3
                # seg_vol_fixed = np.copy(seg)
                # seg_vol_fixed[seg_vol_fixed==4] = 3

                # # "shift" every class by one (as class 4 is now empty)
                # seg_vol_fixed[seg_vol_fixed>4]-=1
            
        
                y_train_batch[vol_num] = np_utils.to_categorical(seg, num_segmask_labels)

            # final batch-wise shuffling (meaningful iff batch_size>1)
            yield shuffle(x_train_batch, y_train_batch)

            gc.collect()

num_sequences = 1

train_generator = vol_generator(vol_list            = training_volumes_path,
                                data_dims           = data_dims,
                                num_segmask_labels  = num_segmask_labels,
                                batch_size          = 1,
                                use_standardisation = use_standardisation,
                                voxelwise_mean      = voxelwise_mean,
                                voxelwise_std       = voxelwise_std,
                                )

# define the validation data as a 5D volumes: VOLNUMxDIM1xDIM2xDIM3xCHs
x_val = np.zeros((validation_set_size, num_slices_sag, num_slices_cor, num_slices_lon, num_sequences),
                 dtype=np.float)

# define the validation GT as a 5D volumes: VOLNUMxDIM1xDIM2xDIM3xN_CLASSES
y_val = np.zeros((validation_set_size, num_slices_sag, num_slices_cor, num_slices_lon, num_segmask_labels),
                 dtype=np.uint8)

vol_num = 0


for vol in validation_volumes_path:

    vol_name = vol.split('/')[-1]

    mri_path = os.path.join(vol, vol_name + anat_suffix)
    segmask_path = os.path.join(vol, vol_name + segm_suffix)

    print('(%03d/%d) - loading validation volume %s and its mask...'%(vol_num+1, validation_set_size, vol_name)),

    # to make the data suitable for pooling, reduce the last dimension so that it can be divided by 2^3
    temp = np.array(nib.load(mri_path).dataobj[:, :, :]).astype(dtype = 'float32')
    
    # check if everything is ok
    if temp.shape != data_dims:
        print('\n Warning: volume "%s" size mismatch: skipping to the next volume...'%(vol_name))
        continue

    if use_standardisation == True:
        temp = volume_zscoring(temp, voxelwise_mean, voxelwise_std)
    
    check_vol_integrity(temp, vol_name)

    x_val[vol_num, :, :, :, 0] = temp


    seg = np.array(nib.load(segmask_path).dataobj[:, :, :]).astype(dtype = 'uint8')

    #  as we don't have many cases of white matter lesions (class 4) merge this class with white matter (class 3
    # seg_vol_fixed = np.copy(seg)
    # seg_vol_fixed[seg_vol_fixed==4] = 3
    
    # # "shift" every class by one (as class 4 is now empty)
    # seg_vol_fixed[seg_vol_fixed>4]-=1
    
    y_val[vol_num] = np_utils.to_categorical(seg, num_segmask_labels)
    
    vol_num = vol_num + 1

    print('Done.')

# to check if the class merge operation worked out as intended, make sure each voxel was
# assigned to exactly one class
# assert np.min(np.sum(y_val, axis = 4)) == 1 

tb_log_dir = os.path.join('../output/tb_logs/' + dataset_name.split('/')[0])

if not os.path.exists(tb_log_dir):    
    os.makedirs(tb_log_dir)

"""
write_histogram = False

if write_histogram == True:
    tb_callback = utils.TrainValTensorBoard(model_name      = model_name,
                                            write_graph     = True, 
                                            write_images    = False,
                                            log_dir         = tb_log_dir,
                                            histogram_freq  = 1,
                                            batch_size      = 1,
                                            )
else:
    tb_callback = utils.TrainValTensorBoard(model_name    = model_name,
                                            write_graph   = True, 
                                            write_images  = False,
                                            log_dir       = tb_log_dir,
                                            )
"""                                
print("Model will be saved at the following location: %s"%(model_path))          
                                          
checkpointer = ModelCheckpoint(model_path, verbose = 1, save_best_only = True)
earlystopper = EarlyStopping(patience = 35, verbose = 1)

# dimension of a single training volume (e.g. t1|ir|t2)
input_data_dims = x_val.shape[1:]

model = models_dict[model_arch](input_data_dims,
                                num_segmask_labels,
                                encoder_act_function = encoder_act_funct,
                                decoder_act_function = decoder_act_funct,
                                classification_act_function = classification_act_funct,
                                loss_function = loss_funct,
                                learning_rate = learning_rate,
                                min_filters_per_layer = num_filters
                                )


# log model architecture on "_ARCH.txt" file
with open(archfile_path, "w") as archfile:
    model.summary(print_fn = lambda x: archfile.write(x + '\n\n')) 


# init log dictionary
log_dict = dict()

start_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

# update dictionary
log_dict.update( dict(start_date        = start_date,
                      model_name        = model_name,
                      num_tr_samples    = training_set_size,
                      tr_samples        = training_subdir_list,
                      num_val_samples   = validation_set_size,
                      val_samples       = validation_subdir_list,
                      enc_act_funct     = encoder_act_funct,
                      dec_act_funct     = decoder_act_funct,
                      class_act_funct   = classification_act_funct,
                      loss_funct        = loss_funct,
                      lr                = learning_rate,
                      num_filters       = num_filters,
                      zscoring          = use_standardisation,
                      zscoring_mean     = voxelwise_mean_path,
                      zscoring_std      = voxelwise_std_path,
                      )
                )


with open(logfile_path, "w") as logfile:
    json.dump(log_dict, logfile, indent = 4, sort_keys = False)

# print time and some info in bash
current_date_time = datetime.now()
print (str(current_date_time))

start_time = time.time()

results = model.fit_generator(generator = train_generator,
                              validation_data = (x_val, y_val),
                              steps_per_epoch = training_samples,
                              epochs = num_epochs,
                              verbose = 1,
                              callbacks = [checkpointer],
                              max_queue_size = 1,
                              use_multiprocessing = False,
                              )
#import pdb;pdb.set_trace()
model.save(os.path.join(models_dir, 'epoch50.h5'))

# once the training finishes, log the elapsed time
elapsed_time = time.time() - start_time

log_dict.update( dict(elapsed_time = elapsed_time) )

with open(logfile_path, "w") as logfile:
    json.dump(log_dict, logfile, indent = 4, sort_keys = False)
