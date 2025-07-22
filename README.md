# Fraud Detection using Machine Learning

## 🎯 Project Overview
Developed a machine learning model to detect fraudulent financial transactions with 99.99% accuracy for Accredian internship assignment.

## 📊 Results
- **Accuracy**: 99.99%
- **Precision**: 100% (Zero false positives)
- **Recall**: 95.24% (Detected 20 out of 21 frauds)
- **F1-Score**: 97.56%
- **Potential Fraud Prevention**: $57.8M

## 🔍 Key Findings
1. Fraud occurs only in TRANSFER and CASH_OUT transactions
2. Fraudsters typically empty accounts completely
3. Balance calculation errors are the strongest fraud indicator (23.5% importance)
4. Fraudulent transactions never go to merchants

## 💻 Technologies Used
- Python 3.x
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning
- SMOTE for handling class imbalance
- Matplotlib & Seaborn for visualization

## 📁 Project Structure
├── Fraud_Detection_Analysis.ipynb # Main analysis notebook
├── models/ # Saved models
├── reports/ # Summary reports
└── requirements.txt # Python dependencies

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook

## 💡 Business Recommendations
1. Implement real-time balance verification
2. Flag accounts being emptied
3. Additional verification for TRANSFER/CASH_OUT > $10,000
4. Monitor accounts with no merchant transactions

## 👤 Author
[Atul Kumar] - [https://www.linkedin.com/in/atul-kumar-6b45652b5/]