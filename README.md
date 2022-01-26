# Residual Channel Attention Network for Brain Glioma Segmentation

This is the official public PyTorch implementation for our paper [Residual Channel Attention Network for Brain Glioma Segmentation],
which was accepted by EMBC

## Dependencies

- Python (>=3.6)
- Pytorch (>=1.3.1)
- opencv-python
- matplotlib
- h5py
- medpy
- scikit-image
- nibabel
- seg_metrics

## Data Preparation

The dataset is obtained from [Multimodal Brain Tumor Segmentation Challenge (BraTS)](https://www.med.upenn.edu/cbica/brats2021/). Place the downloaded dataset in the right directory according to your path in `systemsetup.py` and run the pre-processing code `dataProcessing/brats18_data_loader.py`, `dataProcessing/brats18_validation_data_loader.py`. You will get the `data_3D_size_160_192_160_res_1.0_1.0_1.0.hdf5` and `data_3D.hdf5` for training and validation respectively.

## Training

Set a correct directory path in the `systemsetup.py`. Run

```bash
python train.py
```

## Validation/Testing

Set a correct directory path in the `systemsetup.py`. Uncomment the paramters in your experiments file (here `experiments/canet.py`) and run `train.py`.

```python
VALIDATE_ALL = False
PREDICT = True
RESTORE_ID = YOUR_CKPT_ID
RESTORE_EPOCH = 199
```
```bash
python train.py
```

## Visualize Segmentation Probability Map

Also uncomment the paramter `VISUALIZE_PROB_MAP` in your experiments file (here `experiments/canet.py`) and run `train.py`.

```python
VISUALIZE_PROB_MAP = True
```
```bash
python train.py
```
