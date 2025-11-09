Transaction Fraud Detection using PaySim Dataset

Project Overview
This project aims to detect fraudulent financial transactions using the PaySim Synthetic Financial Transactions dataset. The dataset simulates real-world mobile money transactions, making it suitable for studying fraud detection models and analyzing fraudulent behavior patterns. Although similar to credit card fraud detection, this project focuses on general financial transaction fraud detection.

Dataset
Dataset Link: https://www.kaggle.com/datasets/ealaxi/paysim1

Description:
The PaySim dataset contains simulated mobile money transactions inspired by real-world financial systems. It includes various transaction types and account balance information for both sender and receiver accounts.

Key Features:
type - Type of transaction (CASH_IN, CASH_OUT, TRANSFER, PAYMENT, DEBIT)
amount - Transaction amount
oldbalanceOrg - Balance of the sender before the transaction
newbalanceOrig - Balance of the sender after the transaction
oldbalanceDest - Balance of the receiver before the transaction
newbalanceDest - Balance of the receiver after the transaction
isFraud - Indicates whether the transaction is fraudulent (1 for fraud, 0 for normal)

Project Workflow

1. Data Preprocessing
   - Loaded and explored the PaySim dataset
   - Encoded categorical variables such as type
   - Removed non-numeric columns like nameOrig and nameDest
   - Scaled numerical features using StandardScaler

2. Handling Class Imbalance
   - The dataset is highly imbalanced with very few fraudulent transactions
   - Used SMOTE (Synthetic Minority Oversampling Technique) to balance the training data
   - Created synthetic minority samples for better model learning

3. Model Training
   The following models were trained and compared:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - XGBoost Classifier

4. Model Evaluation
   Each model was evaluated using the following metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC Score

Results
After applying SMOTE, the recall and F1-score for the minority (fraud) class improved significantly. Among all models tested, Random Forest and XGBoost provided the best overall performance. XGBoost achieved the highest recall and AUC score, making it the best model for fraud detection on this dataset.

Tools and Libraries
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
XGBoost
Matplotlib
Seaborn

How to Run
1. Clone the repository
2. Install dependencies using:
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
3. Download the dataset from Kaggle and save it as paysim.csv in the project directory
4. Run the Python script or notebook:
   python fraud_detection.py

Future Work
- Explore deep learning models such as LSTM or Autoencoders
- Implement hybrid sampling techniques like SMOTE + Tomek Links
- Develop a real-time transaction fraud detection system

Conclusion
This project demonstrates how SMOTE-based data balancing and tree-based machine learning models can effectively identify fraudulent transactions in highly imbalanced datasets such as PaySim. The approach achieves strong recall and balanced performance, making it suitable for real-world financial fraud detection scenarios.
