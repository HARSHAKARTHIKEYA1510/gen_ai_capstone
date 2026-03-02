Project Overview

This project builds an AI-driven customer analytics system to predict telecom customer churn and evolve into an intelligent agent-based retention strategist.

Customer churn directly impacts revenue in subscription-based industries. This system:

Predicts churn probability using historical behavioral data

Identifies key churn drivers

Generates structured retention strategies based on risk level

The project is divided into two milestones:

Milestone 1: Classical Machine Learning pipeline for churn prediction

Milestone 2: Agentic AI system with reasoning + RAG-based strategy generation

🛠 Technology Stack
Layer	Technology
ML Models	Logistic Regression, Decision Tree (Scikit-Learn)
Data Processing	Pandas, NumPy, StandardScaler
Evaluation	Accuracy, Precision, Recall, F1-Score, ROC-AUC
UI	Streamlit
Agent Framework (M2)	LangGraph
Vector DB (M2)	Chroma / FAISS
LLMs (M2)	Open-source / Free-tier APIs
📂 Dataset Information

Dataset: Telco Customer Churn

Records: 7,032 customers

Features: 30 (after encoding)

Target: Churn (Yes/No → 1/0)

Key Features

Tenure

Monthly Charges

Total Charges

Contract Type

Internet Service

Payment Method

Senior Citizen Status

Service Subscriptions

🎯 Milestone 1 — ML-Based Churn Prediction
Objective

Build a classical ML pipeline (no LLMs) to identify high-risk customers.

System Architecture

Raw Data
→ Cleaning & Preprocessing
→ One-Hot Encoding
→ Feature Scaling (StandardScaler)
→ Train-Test Split (80/20)
→ Model Training
→ Evaluation
→ Streamlit Deployment

Models Implemented
1️⃣ Logistic Regression (Final Model)

Accuracy: 78.7%

Strong Recall & ROC-AUC

Outputs churn probability

2️⃣ Decision Tree

Accuracy: 72.4%

Lower generalization performance

✅ Selected Model: Logistic Regression

Reason: Better overall performance and stronger recall for churn class.

📈 Evaluation Metrics

Accuracy

Confusion Matrix

Precision

Recall

F1-Score

ROC-AUC

Special focus was placed on Recall for churn class (minimizing false negatives).

💻 Streamlit Application

The deployed UI allows:

Manual customer input

Real-time churn probability prediction

Risk categorization:

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

Structured retention recommendations

Loaded assets:

logistic_model.pkl

scaler.pkl

feature_columns.pkl

🤖 Milestone 2 — Agentic AI Retention Assistant
Objective

Transform the ML model into an intelligent retention strategist that:

Reasons about churn probability

Retrieves best practices via RAG

Generates structured intervention plans

Operates via LangGraph workflow

Planned Agent Workflow

Customer Profile
→ Risk Assessment Node
→ Knowledge Retrieval (RAG)
→ Strategy Planning Node
→ Structured Retention Report

📦 Deliverables (End-Sem)

Public deployment

Agent workflow documentation

Structured retention reports

Complete GitHub repository

5-minute demo video

🧠 Current Retention Logic (Rule-Based Prototype)
🔴 High Risk (>70%)

Discounted long-term contract

Loyalty benefits

Assign customer success representative

🟡 Medium Risk (40–70%)

Personalized offers

Engagement follow-ups

🟢 Low Risk (<40%)

Standard monitoring

This will evolve into a fully agentic reasoning system in Milestone 2.

📁 Project Structure
GEN_AI_CAPSTONE/
│
├── app.py
├── churn.ipynb
├── logistic_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── telco_dataset.csv
├── project-report.tex
├── project-report.pdf
├── requirements.txt
├── README.md
├── .gitignore
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/gen_ai_capstone.git
cd gen_ai_capstone
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
streamlit run app.py
🌍 Deployment (Planned)

Streamlit Cloud

Hugging Face Spaces

Render

📊 Evaluation Criteria
Phase	Weight	Focus
Mid-Sem	25%	ML pipeline, Feature Engineering, UI, Metrics
End-Sem	30%	Agent reasoning, RAG implementation, Deployment
🔮 Future Improvements

Hyperparameter tuning

Feature importance visualization

SHAP explainability

Real-time analytics dashboard

Multi-step agent reasoning workflow
