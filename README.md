1) Download dataset folder on Kaggle : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
and unzip all the files in ./data/raw (from the project root)

2) python -m venv .venv -> create the virtual environnement

3) pip install -r requirements.txt

4) python src\features\build_metadata.py

5) python src\features\prepare_images.py


Annexes:
- Olaf Ronneberger, Philipp Fischer et Thomas Brox (18 May 2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation* : https://arxiv.org/abs/1505.04597v1