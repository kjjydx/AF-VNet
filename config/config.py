import os
from config.augm import train_transform, val_transform
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels, test_images_folder, test_labels_folder, test_images, test_labels
from semseg.data_loader import SemSegConfig


class SemSegMRIConfig(SemSegConfig):
    train_images = [os.path.join(train_images_folder, train_image)
                    for train_image in train_images]
    train_labels = [os.path.join(train_labels_folder, train_label)
                    for train_label in train_labels]

    val_images = [os.path.join(test_images_folder, test_image)
                    for test_image in test_images]
    val_labels = [os.path.join(test_labels_folder, test_label)
                    for test_label in test_labels]
 
    #val_images = None
    #val_labels = None
    do_normalize = True
    #batch_size = 16
    batch_size = 2
    num_workers = 10
    
    #pad_ref = (48, 64, 48)
    pad_ref = (224, 256, 176)


    lr = 0.01
    epochs = 50
    low_lr_epoch = epochs // 3
    val_epochs = epochs // 10
    cuda = True
    num_outs = 3
    #do_crossval = True
    do_crossval = False
    #num_folders = 5
    num_folders = 2
    num_channels = 16
    transform_train = train_transform
    transform_val = val_transform
    #net = "vnet"
    net = "vnet_att"


LEARNING_RATE_REDUCTION_FACTOR = 2
