# Parkinson's Prediction System

A simple speech-based Parkinson's disease (PD) prediction project that extracts voice/audio biomarkers and uses machine learning models to detect patterns associated with Parkinson's. This repository contains data processing, feature extraction, model training, and evaluation code to help explore how acoustic features can assist early detection.

## Table of contents
- [About](#about)
- [Features & Approach](#features--approach)
- [Dataset](#dataset)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Usage examples](#usage-examples)
- [Evaluation & Results](#evaluation--results)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)
- [Contact](#contact)

## About
Parkinson's disease (PD) is a progressive neurological disorder. Changes in voice and speech are among the early and measurable signals of PD. This project demonstrates how to extract acoustic features from voice recordings (e.g., jitter, shimmer, MFCCs) and train classification models to predict PD vs. healthy controls.

## Features & approach
- Preprocessing of audio and tabular voice features
- Feature extraction (statistical + spectral features, optional MFCCs)
- Model training using classical ML models (e.g., Random Forest, SVM) and evaluation pipelines
- Basic experiments and metrics (accuracy, precision, recall, F1, ROC-AUC)
- Notebook(s) for exploratory data analysis and model interpretation

## Dataset
This repo expects voice or speech-based data. Common publicly available datasets used for PD research include:
- UCI Parkinson's Telemonitoring / Parkinson's datasets
- Other voice-recording datasets or your own collected samples

Place your dataset in a folder such as `data/` and follow the preprocessing notebook/script to transform raw inputs into features.

## Getting started

### Requirements
- Python 3.8+
- Common Python packages: numpy, pandas, scikit-learn, librosa (for audio), matplotlib, seaborn
- Optional: jupyterlab or notebook to run the notebooks

Example requirements (create a `requirements.txt` if not present):
numpy
pandas
scikit-learn
librosa
matplotlib
seaborn
joblib

### Install
1. Clone the repo
   git clone https://github.com/Bramhaaa/Parkinsons_predictor.git
2. Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate     # Windows
3. Install dependencies
   pip install -r requirements.txt

### Quick start
- Notebook workflow:
  1. Open `notebooks/` (if present) in Jupyter or VS Code.
  2. Run the EDA notebook to inspect data and run model experiments interactively.

- Script workflow (example commands — replace paths with actual script names in this repo):
  1. Preprocess / extract features:
     python src/preprocess.py --input data/raw --output data/features/features.csv
  2. Train model:
     python src/train.py --features data/features/features.csv --out models/pd_model.pkl
  3. Predict / evaluate:
     python src/predict.py --model models/pd_model.pkl --input data/features/test.csv

If script names differ in this repository, I can update these commands to match the exact filenames.

## Project structure
(Adjust to match the repository)
- data/              # raw and processed datasets
- notebooks/         # analysis and experiments
- src/               # scripts: preprocessing, feature extraction, training, prediction
- models/            # saved model artifacts
- requirements.txt
- README.md

## Usage examples
- Train a classifier and save the model:
  python src/train.py --features data/features/train.csv --model models/rf_model.joblib
- Load a saved model and predict on new samples:
  python src/predict.py --model models/rf_model.joblib --sample data/samples/sample1.wav

## Evaluation & Results
Include a short summary of the best-performing model and evaluation metrics (example placeholders below — replace with real numbers after running experiments):

Best model: Random Forest  
Validation accuracy: 0.92  
Precision: 0.90  
Recall: 0.88  
F1-score: 0.89  
ROC-AUC: 0.95

Consider adding:
- Confusion matrix
- ROC curves
- Feature importance plots
- Cross-validation details

## Contributing
Contributions are welcome! Typical ways to contribute:
- Improve documentation and README
- Add notebooks or scripts for new feature extraction techniques (e.g., deep learning features)
- Improve model evaluation and add baseline comparisons

If you'd like me to open a PR making the README file updated in this repo, I can do that.

## Authors
- Bramha Bajannavar
- Rahul Hongekar

## License
Add a license file (e.g., MIT) if you want the project to be open source. This repo currently does not specify a license — please add LICENSE.md if you want one.

## Contact
For questions or help improving this README or repository structure, open an issue or contact the authors.
