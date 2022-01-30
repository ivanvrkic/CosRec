## Reproducibility study: CosRec, 2D Convolutional Neural Networks for Sequential Recommendation
Reproducibility study for the paper:

*CosRec: 2D Convolutional Neural Networks for Sequential Recommendation, CIKM-2019*

[[arXiv](https://arxiv.org/abs/1908.09972)] [[GitHub](https://github.com/zzxslp/CosRec)]

The code was tested with NVIDIA Tesla P100 with PyTorch 1.1.0 and Python 3.7.12.

## Custom Preprocessing
Custom preprocessing is done by running the following:
```
python preprocess.py --dataset=ml1m
python preprocess.py --dataset=gowalla
```
## Training
To train our model on `ml1m` (with default hyper-parameters): 

```
python train.py --dataset=ml1m
```

or on `gowalla` (change a few hyper-paras based on dataset statistics):

```
python train.py --dataset=gowalla --d=100 --fc_dim=50 --l2=1e-6
```

You should be able to obtain MAPs of ~0.188 and ~0.098 on ML-1M and Gowalla respectively, with the above settings.

## Results evaluation and significance testing
To perform significance testing and evaluation of the results
```
python evaluate.py
```

### Datasets

- Datasets are organized into 2 separate files: **_train.txt_** and **_test.txt_**

- Same as other data format for recommendation, each file contains a collection of triplets:

  > user item rating

  The only difference is the triplets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so we convert them all to 1.

