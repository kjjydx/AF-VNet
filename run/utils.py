import copy
import os
import torch

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from semseg.data_loader import TorchIODataLoader3DTraining
from models.vnet3d import VNet3D
from semseg.utils import zero_pad_3d_image, z_score_normalization


def print_config(config):
    attributes_config = [attr for attr in dir(config)
                         if not attr.startswith('__')]
    print("Config")
    for item in attributes_config:
        attr_val = getattr(config,item)
        if len(str(attr_val)) < 100:
            print("{:15s} ==> {}".format(item, attr_val))
        else:
            print("{:15s} ==> String too long [{} characters]".format(item,len(str(attr_val))))


def check_train_set(config):
    num_train_images = len(config.train_images)
    num_train_labels = len(config.train_labels)

    assert num_train_images == num_train_labels, "Mismatch in number of training images and labels!"

    print("There are: {} Training Images".format(num_train_images))
    print("There are: {} Training Labels".format(num_train_labels))


def check_torch_loader(config, check_net=False):
    train_data_loader_3D = TorchIODataLoader3DTraining(config)
    iterable_data_loader = iter(train_data_loader_3D)
    el = next(iterable_data_loader)
    inputs, labels = el['t1']['data'], el['label']['data']
    print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))
    if check_net:
        net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
        outputs = net(inputs)
        print("Shape of Output: [output {}]".format(outputs.shape))


def print_folder(idx, train_index, val_index):
    print("+==================+")
    print("+ Cross Validation +")
    print("+     Folder {:d}     +".format(idx))
    print("+==================+")
    print("TRAIN [Images: {:3d}]:\n{}".format(len(train_index), train_index))
    print("VAL   [Images: {:3d}]:\n{}".format(len(val_index), val_index))


def print_test():
    print("+============+")
    print("+   Test     +")
    print("+============+")


def train_val_split(train_images, train_labels, train_index, val_index):
    train_images_np, train_labels_np = np.array(train_images), np.array(train_labels)
    train_images_list = list(train_images_np[train_index])
    val_images_list = list(train_images_np[val_index])
    train_labels_list = list(train_labels_np[train_index])
    val_labels_list = list(train_labels_np[val_index])
    return train_images_list, val_images_list, train_labels_list, val_labels_list


def train_val_split_config(config, train_index, val_index):
    train_images_list, val_images_list, train_labels_list, val_labels_list = \
        train_val_split(config.train_images, config.train_labels, train_index, val_index)
    new_config = copy.copy(config)
    new_config.train_images, new_config.val_images = train_images_list, val_images_list
    new_config.train_labels, new_config.val_labels = train_labels_list, val_labels_list
    return new_config


def nii_load(train_image_path):
    train_image_nii = nib.load(str(train_image_path), mmap=False)
    train_image_np = train_image_nii.get_fdata(dtype=np.float32)
    affine = train_image_nii.affine
    return train_image_np, affine


def sitk_load(train_image_path):
    train_image_sitk = sitk.ReadImage(train_image_path)
    train_image_np = sitk.GetArrayFromImage(train_image_sitk)
    origin, spacing, direction = train_image_sitk.GetOrigin(), \
                                 train_image_sitk.GetSpacing(), train_image_sitk.GetDirection()
    meta_sitk = {
        'origin'   : origin,
        'spacing'  : spacing,
        'direction': direction
    }
    return train_image_np, meta_sitk


def nii_write(outputs_np, affine, filename_out):
    outputs_nib = nib.Nifti1Image(outputs_np, affine)
    outputs_nib.header['qform_code'] = 1
    outputs_nib.header['sform_code'] = 0
    outputs_nib.to_filename(filename_out)


def sitk_write(outputs_np, meta_sitk, filename_out):
    outputs_sitk = sitk.GetImageFromArray(outputs_np)
    outputs_sitk.SetDirection(meta_sitk['direction'])
    outputs_sitk.SetSpacing(meta_sitk['spacing'])
    outputs_sitk.SetOrigin(meta_sitk['origin'])
    sitk.WriteImage(outputs_sitk, filename_out)


def np3d_to_torch5d(train_image_np, pad_ref, cuda_dev):
    train_image_np = z_score_normalization(train_image_np)

    inputs_padded = zero_pad_3d_image(train_image_np, pad_ref,
                                      value_to_pad=train_image_np.min())
    inputs_padded = np.expand_dims(inputs_padded, axis=0)  # 1 x Z x Y x X
    inputs_padded = np.expand_dims(inputs_padded, axis=0)  # 1 x 1 x Z x Y x X

    inputs = torch.from_numpy(inputs_padded).float()
    inputs = inputs.to(cuda_dev)
    return inputs


def torch5d_to_np3d(outputs, original_shape):
    outputs = torch.argmax(outputs, dim=1)  # 1 x Z x Y x X
    outputs_np = outputs.data.cpu().numpy()
    outputs_np = outputs_np[0]  # Z x Y x X
    outputs_np = outputs_np[:original_shape[0],:original_shape[1],:original_shape[2]]
    outputs_np = outputs_np.astype(np.uint8)
    return outputs_np


def print_metrics(multi_dices,  multi_jaccards, multi_ppvs, multi_hd95s, f1_scores, train_confusion_matrix):
    multi_dices_np = np.array(multi_dices)
    mean_multi_dice = np.mean(multi_dices_np)
    std_multi_dice = np.std(multi_dices_np, ddof=1)

    multi_jaccards_np = np.array(multi_jaccards)
    mean_multi_jaccard = np.mean(multi_jaccards_np)
    std_multi_jaccard = np.std(multi_jaccards_np, ddof=1)

    multi_ppvs_np = np.array(multi_ppvs)
    mean_multi_ppv = np.mean(multi_ppvs_np)
    std_multi_ppv = np.std(multi_ppvs_np, ddof=1)

    multi_hd95s_np = np.array(multi_hd95s)
    mean_multi_hd95 = np.mean(multi_hd95s_np)
    std_multi_hd95 = np.std(multi_hd95s_np, ddof=1)

    f1_scores = np.array(f1_scores)

    f1_scores_anterior_mean = np.mean(f1_scores[:, 1])
    f1_scores_anterior_std = np.std(f1_scores[:, 1], ddof=1)

    f1_scores_posterior_mean = np.mean(f1_scores[:, 2])
    f1_scores_posterior_std = np.std(f1_scores[:, 2], ddof=1)

    print("+================================+")
    print("Multi Class Dice           ===> {:.4f} +/- {:.4f}".format(mean_multi_dice, std_multi_dice))
    
    print("Multi Class jaccard           ===> {:.4f} +/- {:.4f}".format(mean_multi_jaccard, std_multi_jaccard))
    print("Multi Class ppv           ===> {:.4f} +/- {:.4f}".format(mean_multi_ppv, std_multi_ppv))
    print("Multi Class hd95           ===> {:.4f} +/- {:.4f}".format(mean_multi_hd95, std_multi_hd95))

    print("+================================+")
    print("Images with Dice > 0.8     ===> {} on {}".format((multi_dices_np > 0.8).sum(), multi_dices_np.size))
    print("+================================+")
    print("Hippocampus Anterior Dice  ===> {:.4f} +/- {:.4f}".format(f1_scores_anterior_mean, f1_scores_anterior_std))
    print("Hippocampus Posterior Dice ===> {:.4f} +/- {:.4f}".format(f1_scores_posterior_mean, f1_scores_posterior_std))
    print("+================================+")
    print("Confusion Matrix")
    print(train_confusion_matrix)
    print("+================================+")
    print("Normalized (All) Confusion Matrix")
    train_confusion_matrix_normalized_all = train_confusion_matrix / train_confusion_matrix.sum()
    print(train_confusion_matrix_normalized_all)
    print("+================================+")
    print("Normalized (Row) Confusion Matrix")
    train_confusion_matrix_normalized_row = train_confusion_matrix.astype('float') / \
                                            train_confusion_matrix.sum(axis=1)[:, np.newaxis]
    print(train_confusion_matrix_normalized_row)
    print("+================================+")


def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          already_normalized=False,
                          path_out=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.matshow(cm, cmap=cmap)
    plt.title(title, pad=25.)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize or already_normalized else cm.max() / 2
    print("Thresh = {}".format(thresh))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize or already_normalized:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if path_out is not None:
        plt.savefig(path_out)
    plt.show()
