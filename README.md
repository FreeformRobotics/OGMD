# OGMCD

To address grid map classification problem, we construct a dataset containing **6916** (the labels including 3210 normal and 3706 abnormal )grid maps through an **indoor robot vacuum cleaner**. These grid maps are created with an initial size of **50m×50m**. To further increase the number of training examples, we applied random rotation and offset to cropped areas of **34m×34m** used as training examples. To the best of our knowledge, **OGMCD** is a **large-scale benchmark specifically for indoor grid map classification.**



## Getting Started

### Creating a virtual Python environment using Anaconda

1. Inside of `OGMCD/python/` directory run `conda create -n myenv python=3.6`.
2. Activate the virtual environment by running `source activate myenv`
3. Install requirements from `requirements.txt` by running `pip install -r requirements.txt`

## Dataset training 

We train 400 epochs by Stochastic Gradient Descent (SGD)with the momentum of 0.9 and a weight decay of 1e-4. The learning rate starts from 0.01 and drops every 50 epochs. It takes about 10 hours for the network to converge on an NVIDIA GTX 2080Ti graphics card.

```python
python train.py [OGMCD with train and val folders] train [path to weights file saves] -a [model name]
```

For example

```python
python train.py [OGMCD-folder with train and val folders] train ./model_save/ -a se_resnet32
```

## Dataset test

```python
python test.py [OGMCD with test folders] test [path to weights file] -a [model name]
```

For example

```python
python test.py [OGMCD with test folders] test se_resnet32.pth -a se_resnet32
```

# Dataset DownLoad

You may download  the dataset reported in the paper from Google Drive or the Baidu Netdisk 

| Google Drive  | [Link](https://drive.google.com/file/d/1dumOpdy9nxV0xKt0r-Q0UUej-ydjHi7v/view?usp=sharing) |
| ------------- | ------------------------------------------------------------ |
| Baidu Netdisk | [Link](https://pan.baidu.com/s/1TP43dI6IyGbuB6j9C_Tpxg)      |

Baidu Netdisk eval code：yyvs
