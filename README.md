```markdown
# Visa Approval Prediction System (MLOps)

## Overview
Machine learning system predicting visa approval outcomes with 95% F1-score.
Built with MLOps best practices including experiment tracking, model versioning,
and [coming soon: API deployment, containerization, CI/CD].

## Problem Statement
The Office of Foreign Labor Certification (OFLC) processes 700,000+ visa
applications annually. This ML system identifies applications with high approval
probability, streamlining the review process.

## Dataset
- **Source**: OFLC historical visa application data
- **Size**: 25,480 applications
- **Features**: Education, job experience, region, wage, company size, etc.
- **Target**: Certified vs Denied

## Approach
1. **Data Preprocessing**: Handled missing values, encoded categorical variables,
   addressed class imbalance with SMOTE
2. **Model Training**: Evaluated GBM, Random Forest, and AdaBoost with
   RandomizedSearchCV hyperparameter tuning
3. **Model Selection**: GBM with SMOTE achieved best performance
4. **MLOps**: Production training pipeline (train.py) with MLflow experiment
   tracking, model registration, and feature importance artifacts

## Technical Stack
- **ML**: Python, pandas, scikit-learn, imbalanced-learn
- **MLOps**: MLflow (experiment tracking, model registry)
- [Coming soon: FastAPI, Docker, Kubernetes, CI/CD]

## Results
| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| GBM + SMOTE | 0.95 | 0.94 | 0.96 |
| Random Forest + Undersample | 0.94 | 0.93 | 0.95 |
| AdaBoost + SMOTE | 0.93 | 0.92 | 0.94 |

## Key Features Influencing Approval
1. Prevailing wage (higher → higher approval)
2. Region of employment (Northeast, West have higher rates)
3. Education level (Master's/PhD → higher approval)
4. Full-time position (vs part-time)

## How to Run

### Prerequisites
\`\`\`powershell
# Python 3.10+ required
pip install -r requirements.txt
\`\`\`

### Train Model (all options)
\`\`\`powershell
# Default: GBM with SMOTE oversampling + hyperparameter tuning
python src/train.py

# Specify model and sampling strategy
python src/train.py --model gbm --sampling over
python src/train.py --model rf  --sampling under
python src/train.py --model ada --sampling original

# Custom data path
python src/train.py --data-path C:\data\EasyVisa.csv

# Skip hyperparameter tuning (fast test run)
python src/train.py --no-tune
\`\`\`

### View Experiment Results
\`\`\`powershell
mlflow ui
# Open http://localhost:5000
\`\`\`

## Project Status
- [x] Data preprocessing and EDA
- [x] Model training and evaluation
- [x] Production training pipeline (train.py) with CLI
- [x] MLflow experiment tracking and model registry
- [ ] FastAPI model serving
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Model monitoring dashboard

## Author
Vision 

## License
MIT License
```
