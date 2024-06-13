##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################
# python run/validate.py
# python run/validate.py --dir=logs/no_augm_torchio
# python run/validate.py --dir=logs/no_augm_torchio --write=0
# python run/validate.py --dir=path/to/logs/dir --write=WRITE --verbose=VERBOSE

##########################
# Imports
##########################
import torch
import numpy as np
import os
from sklearn.model_selection import KFold
import argparse
import sys

##########################
# Local Imports
##########################
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from models.vnet3d import VNet3D
from models.vnet3d_att import VNet3D_att
from config.paths import ( train_images_folder, train_labels_folder, train_prediction_folder,
                           train_images, train_labels,test_labels_folder,test_labels,
                           test_images_folder, test_images, test_prediction_folder)
from run.utils import (train_val_split, print_folder, nii_load, sitk_load, nii_write, print_config,
                       sitk_write, print_test, np3d_to_torch5d, torch5d_to_np3d, print_metrics, plot_confusion_matrix)
from config.config import SemSegMRIConfig
from semseg.utils import multi_dice_coeff, multi_jaccard_coeff, multi_ppv_coeff, multi_hd95_coeff
from sklearn.metrics import confusion_matrix, f1_score




def run(logs_dir="logs/lightmunet", write_out=False, plot_conf=False):
    ##########################
    # Config
    ##########################
    config = SemSegMRIConfig()
    print_config(config)

    ###########################
    # Load Net
    ###########################
    cuda_dev = torch.device("cuda")

    # Load From State Dict
    path_net = os.path.join(logs_dir,"model_epoch_0430.pht")
    #path_net = "logs/model_epoch_0490.pht"
    net = VNet3D_att(num_outs=config.num_outs, channels=config.num_channels)
    net.load_state_dict(torch.load(path_net))


    ###########################
    # Eval Loop
    ###########################
    use_nib = True
    plot_conf = True

    #pad_ref = (48,64,48)
    pad_ref = (224, 256, 176)

    multi_dices = list()
    multi_jaccards = list()
    multi_ppvs = list()
    multi_hd95s = list()
    f1_scores = list()
    train_confusion_matrix = np.zeros((config.num_outs, config.num_outs))

    net = net.cuda(cuda_dev)
    net.eval()

    for idx, test_image in enumerate(test_images):
        test_image_path = os.path.join(test_images_folder, test_image)

        if use_nib:
            test_image_np, affine = nii_load(test_image_path)
        else:
            test_image_np, meta_sitk = sitk_load(test_image_path)

        with torch.no_grad():
            inputs = np3d_to_torch5d(test_image_np, pad_ref, cuda_dev)
            outputs = net(inputs)
            outputs_np = torch5d_to_np3d(outputs, test_image_np.shape)

        if write_out:
            filename_out = os.path.join(test_prediction_folder, test_image)
            if use_nib:
                nii_write(outputs_np, affine, filename_out)
            else:
                sitk_write(outputs_np, meta_sitk, filename_out)


        test_label = test_labels[idx]
        test_label_path = os.path.join(test_labels_folder, test_label)
        if use_nib:
            test_label_np, _ = nii_load(test_label_path)
        else:
            test_label_np, _ = sitk_load(test_label_path)






        multi_dice = multi_dice_coeff(np.expand_dims(test_label_np,axis=0),
                                      np.expand_dims(outputs_np,axis=0),
                                      config.num_outs)
        multi_dices.append(multi_dice)

        multi_jaccard = multi_jaccard_coeff(np.expand_dims(test_label_np,axis=0),
                                      np.expand_dims(outputs_np,axis=0),
                                      config.num_outs)
        multi_jaccards.append(multi_jaccard)
        
        multi_ppv = multi_ppv_coeff(np.expand_dims(test_label_np,axis=0),
                                      np.expand_dims(outputs_np,axis=0),
                                      config.num_outs)
        multi_ppvs.append(multi_ppv)
        
        multi_hd95 = multi_hd95_coeff(np.expand_dims(test_label_np,axis=0),
                                      np.expand_dims(outputs_np,axis=0),
                                      config.num_outs)
        multi_hd95s.append(multi_hd95)
        
        print("idx:{} Multi Class Dice Coeff = {:.4f} Jaccard Coeff = {:.4f} PPV Coeff = {:.4f} Hd95 Coeff = {:.4f}".format(idx, multi_dice, multi_jaccard, multi_ppv, multi_hd95))


        f1_score_idx = f1_score(test_label_np.flatten(), outputs_np.flatten(), average=None)
        cm_idx = confusion_matrix(test_label_np.flatten(), outputs_np.flatten())
        train_confusion_matrix += cm_idx
        f1_scores.append(f1_score_idx)


    print_metrics(multi_dices, multi_jaccards, multi_ppvs, multi_hd95s, f1_scores, train_confusion_matrix)

    if plot_conf:
        plot_confusion_matrix(train_confusion_matrix,
                              target_names=None, title='Cross-Validation Confusion matrix',
                              cmap=None, normalize=False, already_normalized=False,
                              path_out="images/conf_matrix_no_norm_no_augm_torchio.png")
        plot_confusion_matrix(train_confusion_matrix,
                              target_names=None, title='Cross-Validation Confusion matrix (row-normalized)',
                              cmap=None, normalize=True, already_normalized=False,
                              path_out="images/conf_matrix_normalized_row_no_augm_torchio.png")


############################
# MAIN
############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Validation for Hippocampus Segmentation")
    parser.add_argument(
        "-V",
        "--verbose",
        default=False, type=bool,
        help="Boolean flag. Set to true for VERBOSE mode; false otherwise."
    )
    parser.add_argument(
        "-D",
        "--dir",
        default="logs/vnet_att", type=str,
        help="Local path to logs dir"
    )
    parser.add_argument(
        "-W",
        "--write",
        default=False, type=bool,
        help="Boolean flag. Set to true for WRITE mode; false otherwise."
    )
    parser.add_argument(
        "--net",
        default='vnet',
        help="Specify the network to use [unet | vnet] ** FOR FUTURE RELEASES **"
    )

    args = parser.parse_args()
    run(logs_dir=args.dir, write_out=args.write, plot_conf=args.verbose)
