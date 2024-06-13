# Hippocampus Segmentation from MRI using AFV-Net


## Usage
Use ```python setup.py install``` for installing this package.
A complete run (dataset download, train, validation) of the package may be the following:
```console

cd AFNet
python setup.py install
python run/download.py
python run/train.py 
python run/validate.py
```
### Dataset
If you want to download the original dataset, run ```run/download.py```.
The syntax is as follows:
```console
python run/download.py --dir=path/to/dataset/dir
```
### Training
If you simply want to perform the training, run ```run/train.py```.
The syntax is as follows:
```console
python run/train.py --epochs=NUM_EPOCHS --batch=BATCH_SIZE --workers=NUM_WORKERS --lr=LR
```
If you want to edit the configuration, you can also modify the ```config/config.py``` file. 
In particular, consider the class ```SemSegMRIConfig```. 
If you want to play with data augmentation (built with ```torchio```), 
modify the ```config/augm.py``` file.

### Validation
If you want to perform the cross-validation, run ```run/validate.py``` or ```run/validate_torchio.py```.
The syntax is as follows:
```console
python run/validate.py --dir=path/to/logs/dir --write=WRITE --verbose=VERBOSE
```
```console
python run/validate_torchio.py --dir=path/to/logs/dir --verbose=VERBOSE
```
The former adopts a loop from scratch, whereas the latter exploits the DataLoader created upon ```torchio```. 

