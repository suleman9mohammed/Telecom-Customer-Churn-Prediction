# Telecom Customer Churn Prediction

[**Dataset Download Link**](https://1024terabox.com/s/1yQYwrqjGFOInsSjaHPd9Tw)

### Project Overview:

This project focuses on building predictive models to identify high-value customers at risk of churning in the telecom industry. The dataset used spans four months of customer data, and the goal is to predict whether a customer will churn in the final month based on data from the first three months. The project aims to reduce churn rates by identifying the main indicators of churn, which will help telecom companies take proactive steps to retain high-value customers.

### Problem Statement:

Customer churn is a significant problem for telecom companies, especially in highly competitive markets where the churn rate can be as high as 25%. Given that it is more cost-effective to retain existing customers than to acquire new ones, the telecom industry places a high priority on customer retention.

In this project, the objective is to predict which high-value customers (those contributing the most to revenue) are likely to churn, and to identify the key factors that influence customer churn. The dataset focuses on the prepaid segment of the telecom market in India and Southeast Asia, where churn prediction is more critical due to the absence of formal termination of services.

### Dataset:

The dataset contains customer-level data for four months (June, July, August, and September). The objective is to predict churn in the last month (September) using the features from the first three months (June, July, August).

Key variables include:

- **Customer usage**: Incoming and outgoing calls (local and international), mobile internet usage (2G and 3G), and other telecom services.
- **Recharge information**: Data on recharges made by customers, including amounts and recharge types.
- **Customer segmentation**: High-value customers, defined based on the 70th percentile of average recharge amount in the first two months.
- **Churn indicator**: Binary value indicating whether a customer has churned in the fourth month.

### Business Objective:

The business objective is twofold:

1. **Predict churn**: Develop a model to predict whether high-value customers are likely to churn in the near future. This will allow the telecom company to take corrective actions (e.g., offering special discounts or plans) to retain these customers.
2. **Identify churn indicators**: Determine the key variables that are strong predictors of churn, helping the business understand the underlying reasons for customer churn.

### Approach:

1. **Data Preprocessing**:
   - Convert columns to appropriate formats.
   - Handle missing values and outliers.
   - Create new features based on business understanding, which can help in predicting churn.
   
2. **Filter High-Value Customers**:
   - Define high-value customers based on recharge amounts in the first two months (good phase). Customers with a recharge amount greater than or equal to the 70th percentile are classified as high-value.

3. **Tagging Churners**:
   - A customer is tagged as a churner (churn=1) if they have not made any calls or used mobile internet in the churn month (September).
   - Remove all columns corresponding to the churn month after tagging churners.

4. **Exploratory Data Analysis (EDA)**:
   - Analyze customer usage patterns, recharge behavior, and other features to identify trends.
   - Use visualizations to understand key drivers of churn.

5. **Dimensionality Reduction**:
   - Apply Principal Component Analysis (PCA) to reduce the number of variables while retaining most of the variance.
   
6. **Model Building**:
   - Build multiple models (e.g., Logistic Regression, Decision Trees, Random Forest) to predict churn.
   - Use class imbalance techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or weighted models to handle the imbalance in churn vs. non-churn customers.
   - Tune hyperparameters to improve model performance.

7. **Model Evaluation**:
   - Evaluate models using appropriate metrics (e.g., precision, recall, F1-score, AUC-ROC) with a focus on accurately identifying churners.
   - Choose the best-performing model based on these metrics.

8. **Identifying Important Features**:
   - Use models like Logistic Regression or Decision Trees to identify the most important features that predict churn.
   - Visualize and interpret the key features.

9. **Recommendation**:
   - Provide recommendations to the business on strategies to reduce churn, based on the insights from the model.

### Requirements:

- **Python** (v3.6+)
- **Jupyter Notebook**
- Libraries used in this project:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn (for PCA, classification models, and metrics)
  - imbalanced-learn (for handling class imbalance)

### Instructions to Run:

1. Download or clone the repository from [GitHub Link].
2. Open the Jupyter notebook (`telecom_churn_prediction.ipynb`).
3. Ensure all necessary libraries are installed by running `pip install -r requirements.txt`.
4. Run the notebook step by step to preprocess the data, build and evaluate the models, and derive insights on churn prediction.
5. Review the outputs and recommendations provided at the end of the notebook.

### Files:

- **Jupyter Notebook**: Contains the complete code for data preparation, model building, evaluation, and feature analysis.
- **Dataset**: Contains customer data for four months, with columns representing different usage and recharge features.
- **Data Dictionary**: A document explaining the abbreviations and meanings of various columns in the dataset.

### Evaluation Metrics:

- **Precision**: Measures the percentage of predicted churners that are actual churners.
- **Recall**: Measures the percentage of actual churners that are correctly identified by the model.
- **F1-score**: Harmonic mean of precision and recall, used as a balanced metric for model performance.
- **AUC-ROC Curve**: Measures the model's ability to distinguish between churners and non-churners.

### Repository Name Suggestions:

- `telecom-customer-churn-prediction`
- `telecom-churn-analysis`
- `customer-churn-prediction`
- `telecom-high-value-churn-prediction`

### Conclusion:

This project uses customer data from the telecom industry to predict churn among high-value customers. By identifying the key factors driving churn, the model will help telecom companies develop targeted retention strategies. The use of advanced machine learning techniques and appropriate evaluation metrics ensures that the project meets its business objectives effectively.
