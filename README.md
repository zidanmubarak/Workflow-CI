# Kriteria 3: Workflow CI dengan MLflow Project

**Dataset**: Climate Change Impact on Agriculture 2024  
**Author**: Zidan Mubarak  

[![MLflow CI/CD - Advanced](https://github.com/zidanmubarak/Workflow-CI/actions/workflows/mlflow-ci.yml/badge.svg)](https://github.com/zidanmubarak/Workflow-CI/actions/workflows/mlflow-ci.yml)

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml                 # GitHub Actions workflow (Advanced)
├── MLProject/
│   ├── MLproject                         # MLflow project configuration
│   ├── conda.yaml                        # Conda environment
│   ├── modelling.py                      # Model training script
│   ├── climate_change_preprocessing.csv  # Dataset preprocessing
│   └── docker_hub_link.txt              # Docker Hub repository link
├── requirements.txt                      # Python dependencies
└── README.md                            # Dokumentasi ini
```

## 🎯 Kriteria yang Dipenuhi

1. **Folder MLProject** ✅
   - MLproject file dengan parameter configuration
   - conda.yaml untuk environment management
   - modelling.py untuk training
   - Dataset preprocessing

2. **Workflow CI dengan GitHub Actions** ✅
   - Automatic trigger on push/PR
   - MLflow Project execution
   - Model training otomatis

3. **Artifact Storage** ✅
   - Upload artifacts ke GitHub Actions
   - Model files (pickle)
   - Visualizations (confusion matrix, feature importance)
   - MLflow runs data
   - Retention: 30 days

4. **Docker Integration** ✅
   - Build Docker image dengan `mlflow build-docker`
   - Push ke Docker Hub
   - Tagged dengan version number
   - Fallback custom Dockerfile

## 🚀 Cara Menggunakan

### 1. Setup Repository di GitHub

```bash
# Clone atau create new repository
git clone https://github.com/[username]/Workflow-CI.git
cd Workflow-CI

# Atau jika membuat baru
git init
git remote add origin https://github.com/[username]/Workflow-CI.git
```

### 2. Setup GitHub Secrets

Tambahkan secrets di GitHub repository:

1. **DOCKER_USERNAME**: Username Docker Hub Anda
2. **DOCKER_PASSWORD**: Password atau Access Token Docker Hub

**Cara menambahkan secrets:**
1. Buka repository di GitHub
2. Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Tambahkan DOCKER_USERNAME dan DOCKER_PASSWORD

### 3. Push ke GitHub

```bash
git add .
git commit -m "Setup MLflow CI/CD pipeline"
git push origin main
```

Workflow akan otomatis berjalan!

### 4. Monitor Workflow

1. Buka repository di GitHub
2. Click tab "Actions"
3. Lihat workflow run yang sedang berjalan
4. Check logs untuk setiap step

### 5. Download Artifacts

Setelah workflow selesai:
1. Buka workflow run di tab Actions
2. Scroll ke bawah ke bagian "Artifacts"
3. Download `model-artifacts-[run-number]`

### 6. Pull Docker Image

```bash
# Pull image dari Docker Hub
docker pull [username]/climate-change-ml:latest

# Run container
docker run -p 5000:5000 [username]/climate-change-ml:latest
```

## 📊 Workflow Steps (Advanced)

Workflow CI mencakup tahapan berikut:

1. **Setup Job** ✅
   - Checkout code
   - Setup Python 3.10

2. **Install Dependencies** ✅
   - Install MLflow dan dependencies
   - Setup environment

3. **Set MLflow Tracking URI** ✅
   - Configure MLflow tracking

4. **Run MLflow Project** ✅
   - Execute model training
   - Log metrics dan parameters

5. **Get Latest MLflow Run ID** ✅
   - Retrieve run information

6. **Collect Artifacts** ✅
   - Model files
   - Visualizations
   - MLflow runs

7. **Upload to GitHub** ✅
   - Upload artifacts ke GitHub Actions
   - 30 days retention

8. **Build Docker Image** ✅
   - Using `mlflow build-docker`
   - Fallback custom Dockerfile

9. **Log in to Docker Hub** ✅
   - Authenticate dengan secrets

10. **Tag Docker Image** ✅
    - Tag dengan `latest`
    - Tag dengan run number

11. **Push Docker Image** ✅
    - Push ke Docker Hub repository

## 🔧 MLflow Project Configuration

### MLproject File

```yaml
name: Climate_Change_Agriculture_ML

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: string, default: "climate_change_preprocessing.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
      test_size: {type: float, default: 0.2}
      random_state: {type: int, default: 42}
    command: "python modelling.py ..."
```

### Parameters

- **data_path**: Path ke dataset preprocessing
- **n_estimators**: Jumlah trees di Random Forest
- **max_depth**: Maximum depth of trees
- **test_size**: Ratio test set (default: 0.2)
- **random_state**: Random seed (default: 42)

## 🐳 Docker Integration

### Build Docker Image Locally

```bash
cd MLProject

# Method 1: Using MLflow (recommended)
mlflow models build-docker \
  --model-uri "runs:/<run_id>/sklearn_model" \
  --name "climate-change-ml"

# Method 2: Using custom Dockerfile
docker build -t climate-change-ml .
```

### Run Docker Container

```bash
# Run MLflow model server
docker run -p 5000:5000 climate-change-ml

# Test prediction
curl -X POST http://localhost:5000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

### Push to Docker Hub

```bash
# Tag image
docker tag climate-change-ml [username]/climate-change-ml:latest

# Login to Docker Hub
docker login

# Push image
docker push [username]/climate-change-ml:latest
```

## 📈 Model Details

### Dataset
- **Source**: climate_change_preprocessing.csv
- **Features**: Numeric features dari climate change data
- **Target**: Binary classification (High/Low Crop Yield)
- **Split**: 80% training, 20% testing

### Model
- **Algorithm**: Random Forest Classifier
- **Default Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42

### Metrics Logged
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC AUC Score

### Artifacts Generated
1. **Model**: random_forest_model.pkl
2. **Confusion Matrix**: confusion_matrix.png
3. **Feature Importance**: feature_importance.png
4. **Classification Report**: classification_report.txt

## 💡 Tips & Troubleshooting

### GitHub Secrets
- Pastikan DOCKER_USERNAME dan DOCKER_PASSWORD sudah di-set
- Test Docker login manual terlebih dahulu
- Gunakan Access Token instead of password untuk security

### MLflow Project
- Pastikan conda.yaml dependencies lengkap
- Test local execution sebelum push ke GitHub
- Check MLflow tracking URI configuration

### Docker Build
- Workflow menggunakan fallback jika mlflow build-docker gagal
- Custom Dockerfile akan dibuat otomatis
- Image size bisa besar, optimize jika perlu

### Artifacts
- Artifacts disimpan 30 hari di GitHub Actions
- Download sebelum expired jika perlu
- Model bisa di-load dari pickle file

## 📝 Testing Locally

### Test MLflow Project

```bash
cd MLProject

# Run MLflow project
mlflow run . \
  --env-manager=local \
  -P data_path=climate_change_preprocessing.csv \
  -P n_estimators=100 \
  -P max_depth=10
```

### Test Model Training

```bash
cd MLProject

# Run directly
python modelling.py \
  --data-path climate_change_preprocessing.csv \
  --n-estimators 100 \
  --max-depth 10
```
