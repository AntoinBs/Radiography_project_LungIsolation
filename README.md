# 🫁 Lung Pathology Classification & Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

*AI-powered chest X-ray analysis for COVID-19 and lung pathology detection*

[🚀 Demo](https://lungdx.fr)

</div>

---

## 🎯 Overview
<div align='center'>
This project implements a AI pipeline for chest X-ray lung segmentation.
It comes in combination with the previous project that consisting in X-ray lung classification to detect COVID-19, viral pneumonia, lung opacity, and normal cases.
In this repo, you will find the code that trains the segmentation model, and also an fastAPI API which makes the full AI pipeline functionnal.
</div>

### ✨ Key Features

- 🎯 **Dual-Model Architecture**: U-Net for lung segmentation + Xception for classification
- 🔍 **Grad-CAM Visualization**: Explainable AI showing areas of focus
- 🚀 **FastAPI Backend**: API with session management
- 🐳 **Docker Ready**: Containerized deployment
- 🌐 **Website**: https://lungdx.fr/


---

## 🏗️ Architecture

U-Net by Ronneberger et al. ([Paper](https://arxiv.org/abs/1505.04597))

## 🔧 Models Performance

| Model | Task | Accuracy | Precision | Recall | F1-Score | IoU
|-------|------|----------|-----------|---------|----------|----------|
| U-Net | Lung Segmentation | 99.5% | / | / | / | 98%
| Xception | Pathology Classification | 87% | 87% | 88% | 87% | /

### 📊 Classification Results


| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| COVID-19 | 0.76 | 0.94 | 0.84 | 519 |
| Lung Opacity | 0.88 | 0.86 | 0.87 | 885 |
| Normal | 0.96 | 0.86 | 0.91 | 1562 |
| Viral Pneumonia | 0.83 | 1.00 | 0.90 | 209 |

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.9+
- Docker (optional)

### 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AntoinBs/Radiography_project_LungIsolation.git
   cd Radiography_project_LungIsolation
   ```

2. **Set up environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download dataset**
   ```bash
   # Download from Kaggle: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
   # Unzip to ./data/raw/
   ```

4. **Prepare data and train models**
   ```bash
   python src/features/build_metadata.py
   python src/features/prepare_images.py
   python -m src.models.train_model
   ```

5. **Start the API**
   ```bash
   # Firstly train the classification model : https://github.com/AntoinBs/Radiography_project_LungDisease   
   # Then put it in the 'models' folder
   uvicorn src.app.app:app --reload --port 8000
   ```


### 🐳 Docker Deployment
⚠️ Firstly train the classification model : https://github.com/AntoinBs/Radiography_project_LungDisease and put it in the "models" folder
```bash
# Build the image
docker build -f Dockerfile_non_prod -t lung-pathology-api .

# Run the container
docker run -p 8000:8000 lung-pathology-api
```

---

## 📚 API Documentation

### 🔗 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload/image/` | POST | Upload chest X-ray image |
| `/predict/segmentation/{session_id}` | POST | Generate lung segmentation |
| `/predict/classification/{session_id}` | POST | Classify pathology |
| `/result/segmentation/{session_id}` | GET | Get segmentation mask |
| `/result/classification/{session_id}` | GET | Get Grad-CAM heatmap |
| `/result/prediction/{session_id}` | GET | Get prediction results |

---

## 🔬 Technical Details

### 🧠 Model Architecture

**Segmentation Model (U-Net)**
- Encoder: Downsampling like U-Net arhitecture, beginning with 16 filters in the first layer
- Decoder: Upsampling + Skip connections
- Loss: Binary Cross-entropy
- Optimizer: Adam (lr=1e-4)


### 📊 Dataset

- **Source**: [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- **Classes**: 4 (COVID-19, Lung Opacity, Normal, Viral Pneumonia)
- **Total Images**: 21,165
- **Split**: 70% Train, 15% Validation, 15% Test

---

## 📈 Results

### 🎯 Segmentation Quality
- **IoU**: 0.98
- **Accuracy**: 99,5%

---

## 📞 Contact

- 👨‍💻 **Author**: [Antoine Bas](https://github.com/yourusername)
- 📧 **Email**: antoine.bas@hotmail.fr
- 💼 **LinkedIn**: [Antoine Bas](https://www.linkedin.com/in/antoine-bas/)

---

<div align="center">

**⭐ Star this repo if you found it interesting! ⭐**

[🔝 Back to top](#-lung-pathology-classification--segmentation)

</div>
