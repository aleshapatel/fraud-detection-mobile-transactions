# Fraud Detection in Mobile Financial Transactions

## About the Project
This project uses Machine Learning to identify potentially fraudulent mobile financial transactions. It analyses transaction details and uses an XGBoost classification model to predict whether a transaction is legitimate or fraudulent.

## Objectives
- Analyse mobile financial transaction data.
- Identify patterns associated with fraudulent transactions.
- Train a Machine Learning classification model.
- Predict potentially fraudulent transactions.
- Present insights through a Power BI dashboard.

## Technologies Used
- Python
- Pandas
- XGBoost
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Power BI

## Machine Learning
The Python training script uses the XGBoost classifier.

The project also includes a Jupyter Notebook for analysis and experimentation.

## Features
- Transaction data analysis
- Feature engineering
- XGBoost model training
- Fraud prediction
- Interactive Streamlit application
- Power BI dashboard

## Project Files
- `Fraud_Analysis_Dataset(in).csv` – Transaction dataset
- `Fraud_Detection_ML (1).ipynb` – Jupyter Notebook
- `main.py` – Model training and evaluation
- `app.py` – Streamlit application
- `fraud_model.joblib` – Trained model
- `feature_names.joblib` – Model feature information
- `dashboard.pbix` – Power BI dashboard
- `Fraud_Detection.pptx` – Project presentation
- `requirements.txt` – Python dependencies

## Installation and Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/student-trcac14/fraud-detection-mobile-transactions.git
   ```

2. Open the project folder:

   ```bash
   cd fraud-detection-mobile-transactions/fraud-detection-mobile-transactions
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Project

To train the model:

```bash
python main.py
```

To launch the Streamlit application:

```bash
streamlit run app.py
```

## Author
**Alesha Patel**

B.Sc. Computer Science  
University of Mumbai  
Expected Graduation: 2027
