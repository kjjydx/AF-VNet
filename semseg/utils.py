import numpy as np
from medpy.metric.binary import hd95

# from config.paths import train_images_folder, train_labels_folder


def dice_coeff(gt, pred, eps=1e-5):
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def multi_dice_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        dice_ = dice_coeff(outputs_, labels_)
        dices.append(dice_)
    return sum(dices) / (num_classes-1)


def jaccard_coeff(gt, pred, eps=1e-5):
    pred = pred.astype(np.bool_).astype(np.int_)
    gt = gt.astype(np.bool_).astype(np.int_)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return (intersection + eps) / (union + eps)

def multi_jaccard_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    jaccards = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        jaccard_ = jaccard_coeff(outputs_, labels_)
        jaccards.append(jaccard_)
    return sum(jaccards) / (num_classes-1)



def ppv_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    pred = pred.astype(np.bool_).astype(np.int_)
    gt = gt.astype(np.bool_).astype(np.int_)
    intersection = (pred * gt).sum()
    return (intersection + eps) / (pred.sum() + eps)

def multi_ppv_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    ppvs = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        ppv_ = ppv_coeff(outputs_, labels_)
        ppvs.append(ppv_)
    return sum(ppvs) / (num_classes-1)



def multi_hd95_coeff(gt, pred, num_classes):
    if np.sum(np.unique(pred))==0 or np.sum(np.unique(gt)) == 0:
        return 1.0
    else:
        return hd95(pred, gt)





def one_hot_encode_np(label, num_classes):
    """ Numpy One Hot Encode
    :param label: Numpy Array of shape BxHxW or BxDxHxW
    :param num_classes: K classes
    :return: label_ohe, Numpy Array of shape BxKxHxW or BxKxDxHxW
    """
    assert len(label.shape) == 3 or len(label.shape) == 4, 'Invalid Label Shape {}'.format(label.shape)
    label_ohe = None
    if len(label.shape) == 3:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]))
    elif len(label.shape) == 4:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2], label.shape[3]))
    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)
    return label_ohe


def min_max_normalization(input):
    return (input - input.min()) / (input.max() - input.min())


def z_score_normalization(input):
    input_mean = np.mean(input)
    input_std = np.std(input)
    # print("Mean = {:.2f} - Std = {:.2f}".format(input_mean,input_std))
    return (input - input_mean)/input_std


def zero_pad_3d_image(image, pad_ref=(64,64,64), value_to_pad = 0):
    if value_to_pad == 0:
        image_padded = np.zeros(pad_ref)
    else:
        image_padded = value_to_pad * np.ones(pad_ref)
    image_padded[:image.shape[0],:image.shape[1],:image.shape[2]] = image
    return image_padded
