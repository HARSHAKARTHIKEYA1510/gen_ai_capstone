Project 5: Customer Churn Prediction & Agentic Retention Strategy
From Predictive Analytics to Intelligent Intervention
Project Overview

This project presents the design and implementation of an AI-driven customer analytics system that predicts telecom customer churn and evolves toward an intelligent, agent-based retention strategist.

Customer churn significantly impacts revenue in subscription-based industries. This system leverages classical machine learning techniques to:

Predict churn probability using historical behavioral data

Identify key churn drivers

Provide structured retention strategies based on risk levels

The project is structured into two milestones:

Milestone 1: Classical machine learning techniques applied to historical telecom data for churn prediction.

Milestone 2: Extension into an agentic AI system capable of reasoning about churn risk, retrieving best practices using RAG (Retrieval-Augmented Generation), and generating structured intervention plans.

Constraints & Requirements

Team Size: 3–4 Students

API Budget: Free Tier Only (Open-source models / Free APIs)

Agent Framework (M2): LangGraph

Hosting: Mandatory (Streamlit Cloud / Hugging Face Spaces / Render)

Technology Stack
Component	Technology
ML Models (M1)	Logistic Regression, Decision Tree (Scikit-Learn)
Data Processing	Pandas, NumPy, StandardScaler
Evaluation Metrics	Accuracy, Precision, Recall, F1-Score, ROC-AUC
UI Framework	Streamlit
Agent Framework (M2)	LangGraph (Planned)
Vector Database (M2)	Chroma / FAISS (Planned)
LLMs (M2)	Open-source / Free-tier APIs (Planned)
Dataset Description

Dataset: Telco Customer Churn Dataset
Records: 7,032 customers
Features: 30 (after encoding)
Target Variable: Churn (Yes/No → 1/0)

Key Features:

Tenure

Monthly Charges

Total Charges

Contract Type

Internet Service

Payment Method

Senior Citizen Status

Service Subscriptions

Milestones & Deliverables
Milestone 1: ML-Based Churn Prediction (Mid-Sem)
Objective

Identify customers at risk using historical behavioral data through a classical machine learning pipeline (no LLMs).

System Architecture (M1)
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
One-Hot Encoding
     ↓
Feature Scaling (StandardScaler)
     ↓
Train-Test Split (80/20)
     ↓
Model Training
     ↓
Evaluation & UI Deployment (Streamlit)
Model Implementation

Two classification models were implemented and evaluated:

1. Logistic Regression (Final Selected Model)

Accuracy: 78.7%

Better recall & ROC-AUC compared to Decision Tree

Provides probability outputs for churn risk scoring

2. Decision Tree

Accuracy: 72.4%

Lower generalization performance

Final Model Selected: Logistic Regression
Reason: Superior overall performance and better recall for churn class.

Model Evaluation Metrics

Accuracy

Confusion Matrix

Precision

Recall

F1-Score

ROC-AUC Score

Special emphasis was placed on Recall for the churn class, since false negatives represent lost customers.

Working Application (Streamlit UI)

The system includes an interactive interface that allows:

Manual customer input

Real-time churn probability prediction

Risk categorization:

Low Risk

Medium Risk

High Risk

Structured retention recommendations

The application loads:

Trained Logistic Regression model (.pkl)

Feature scaler

Feature column alignment file

Milestone 2: Agentic AI Retention Assistant (End-Sem)
Objective

Extend the system into an intelligent agent that:

Reasons about churn risk

Retrieves retention best practices using RAG

Generates structured intervention plans

Operates as a multi-state workflow using LangGraph

Planned Agent Workflow
Input Customer Profile
        ↓
Risk Assessment Node
        ↓
Knowledge Retrieval (RAG)
        ↓
Strategy Planning Node
        ↓
Structured Retention Report Generation
Key Deliverables (M2)

Publicly deployed application

Agent workflow documentation (States & Nodes)

Structured retention report generation

Complete GitHub repository

5-minute demo video

Retention Strategy Logic (Current Rule-Based Prototype)

For Milestone 1, retention suggestions are rule-driven:

High Risk (>70%)

Offer discounted long-term contract

Provide loyalty benefits

Assign customer success representative

Medium Risk (40–70%)

Personalized promotional offers

Engagement follow-ups

Low Risk (<40%)

Standard monitoring

This logic will evolve into an intelligent agent-based reasoning system in Milestone 2.

Project Structure
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
Installation & Setup
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/gen_ai_capstone.git
cd gen_ai_capstone

Hosted on GitHub

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
3. Install Dependencies
pip install -r requirements.txt
4. Run Application
streamlit run app.py
Deployment

The application will be publicly hosted using:

Streamlit Cloud (Planned)

Hugging Face Spaces (Optional)

Render (Optional)

Evaluation Criteria
Phase	Weight	Criteria
Mid-Sem	25%	ML technique application, Feature Engineering, UI Usability, Evaluation Metrics
End-Sem	30%	Reasoning quality, RAG implementation, Output clarity, Deployment success
Future Improvements

Hyperparameter tuning

Feature importance visualization dashboard

SHAP explainability integration

Real-time business analytics dashboard

Agentic multi-step reasoning workflow (LangGraph)
