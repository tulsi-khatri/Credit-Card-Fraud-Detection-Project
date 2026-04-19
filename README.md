# Credit Card Fraud Detection using Machine Learning

-> Project Overview:
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, so both supervised and unsupervised approaches are used to analyze and compare performance.

-> Objective:
To build a model that can identify fraudulent transactions accurately and compare the effectiveness of Logistic Regression and Isolation Forest on imbalanced data.

-> Dataset:
The dataset contains credit card transactions with features representing transaction details.

Note: The dataset file is not included in this repository due to its large size (150 MB). It can be accessed separately via a provided link or local source.


-> Technologies Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

-> Machine Learning Models:

- Logistic Regression
A supervised learning model used as a baseline classifier for fraud detection.

- Isolation Forest
An unsupervised anomaly detection model used to identify rare fraudulent transactions.

-> Workflow:
1. Load and explore dataset  
2. Check missing values and class distribution  
3. Standardize features using StandardScaler  
4. Split data into training and testing sets  
5. Train Logistic Regression model  
6. Train Isolation Forest model  
7. Evaluate both models using:
   - Accuracy
   - Confusion Matrix
   - Classification Report  

-> Results Summary:
- Logistic Regression performs well as a baseline model but struggles with imbalanced data.
- Isolation Forest performs better in detecting anomalies and rare fraud cases.

-> Conclusion:
Fraud detection is a challenging problem due to highly imbalanced data. Unsupervised anomaly detection methods like Isolation Forest provide better results compared to basic supervised models in this case. Future improvements can include advanced models like XGBoost, deep learning approaches, and imbalance handling techniques such as SMOTE.

-> How to Run:

1. Install required libraries:
   pip3 install pandas numpy matplotlib scikit-learn
   
2. Run the Python script:
   python credit.py

3. Output will show:
   - Data analysis results
   - Model training logs
   - Evaluation metrics (accuracy, confusion matrix, classification report)
   - Graphs for class distribution


-> Author:
Tulsi Khatri 
BTech student | AI-ML Engineer
