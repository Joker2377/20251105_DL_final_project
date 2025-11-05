# Guide
## Setup environment
```
python -m venv venv
.\venv\Script\activate
```
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
```
pip install -r requirements.txt
```
## Download dataset
[Dataset](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss)
>Download BCSS_512 and extract in directory
 
## Training
1. Open utils.py
    * setting up the variable for training
2. python train.py
3. Check models/ for results