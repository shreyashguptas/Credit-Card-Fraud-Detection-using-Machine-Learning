# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting credit card fraud using machine learning techniques. The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

## Dataset

The dataset can be found on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

# Data Dictionary

| Column | Description |
|--------|-------------|
| Time | Seconds elapsed between each transaction and the first transaction in the dataset. |
| V1-V28 | Principal components obtained through PCA on the original features. Due to confidentiality, the exact nature of these components is not disclosed. |
| Amount | The transaction amount. Can be used for cost-sensitive analysis or to identify patterns related to transaction amounts. |
| Class | The response variable indicating the legitimacy of the transaction (1 for fraud, 0 otherwise). |

## Dataset Characteristics

- Contains transactions made by credit cards in September 2013 by European cardholders.
- Comprises 284,807 transactions over two days, with 492 fraudulent cases.
- The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

## Evaluation Recommendations

- Given the class imbalance, traditional accuracy metrics may not be meaningful.
- It's recommended to use metrics like the Area Under the Precision-Recall Curve (AUPRC).
- Focus on precision and recall to evaluate the performance of fraud detection models.

## Implications for Analysis

### Data Preprocessing
- Consider techniques to handle the imbalanced dataset, such as resampling methods.
- Feature scaling may be necessary for the Amount and Time variables if algorithms used are sensitive to feature scales.

### Modeling Approaches
- Algorithms that perform well with imbalanced data, such as anomaly detection models, might be appropriate.
- Ensemble methods like Random Forests or Gradient Boosting Machines can also be effective.
- Incorporate cross-validation to ensure the robustness of your model.

## Code Inspiration

The code in this project is inspired by the following Kaggle notebook: [Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

## Improvements

Several improvements have been made to the original code:

1. **Torch instead of Tensorflow**: The project now uses the Torch library instead of Tensorflow due to its reliability and strong community support.

2. **Conda Environment**: A `.conda` environment has been set up, and an `environment.yml` file is provided. This allows anyone to easily create the same environment and run the code without any issues. To create the environment, follow these steps:
   - Install Anaconda or Miniconda on your system.
   - Open a terminal or command prompt.
   - Navigate to the project directory.
   - Run the following command:
     ```
     conda env create -f environment.yml
     ```
   - Once the environment is created, activate it using:
     ```
     conda activate credit-card-fraud-detection
     ```

3. **Outlier Removal**: An additional model has been implemented to investigate the impact of outlier removal on the oversampled dataset. The model removes outliers using the Z-score method and evaluates if the accuracy on the test set improves.

## Project Overview

The main goal of this project is to build a machine learning model that can effectively detect fraudulent credit card transactions. The dataset contains various features, including time, amount, and anonymized numerical features (V1-V28) obtained through PCA transformation.

The project follows these steps:

1. **Data Exploration**: The dataset is loaded, and exploratory data analysis is performed to gain insights into the distribution of fraudulent and non-fraudulent transactions.

2. **Data Preprocessing**: The data is preprocessed by scaling the features and splitting it into training and testing sets.

3. **Handling Imbalanced Data**: Since the dataset is highly imbalanced, various techniques are employed to handle the class imbalance, such as undersampling and oversampling (SMOTE).

4. **Model Training and Evaluation**: Different machine learning models, including logistic regression, support vector machines, and neural networks, are trained on the preprocessed data. The models are evaluated using appropriate metrics for imbalanced datasets, such as precision, recall, and F1-score.

5. **Outlier Removal**: An additional experiment is conducted to investigate the impact of outlier removal on the oversampled dataset. Outliers are removed using the Z-score method, and the model's accuracy is evaluated to see if it improves.

## Conclusions

This analysis highlights the effectiveness of various techniques in enhancing fraud detection for imbalanced datasets:

1. **Best Performing Model**: The Logistic Regression classifier trained on SMOTE-oversampled data achieved the highest accuracy of 98.56% on the test set.
   - This outperformed the undersampling approach, which achieved 94.71% accuracy.

2. **Neural Network Performance**:
   - The neural network model trained on SMOTE data showed excellent results.
   - After outlier removal, it achieved 99.93% accuracy, a slight improvement of 0.01 percentage points.

3. **Fraud Detection Metrics**:
   - The SMOTE-based neural network demonstrated high effectiveness in identifying fraud:
     - Recall: 93.81% (106 out of 113 fraud cases correctly identified)
     - Precision: 88.33% (106 true positives out of 120 predicted positives)

4. **Key Insights**:
   - Addressing class imbalance is crucial for improving model performance.
   - Careful feature engineering plays a significant role in fraud detection tasks.

5. **Real-world Application**:
   - It's important to note that in practical scenarios, model performance should be continuously monitored and updated.
   - This ongoing process ensures adaptation to evolving fraud patterns.

These results demonstrate the potential of machine learning in credit card fraud detection while emphasizing the need for dynamic, adaptive approaches in real-world implementations.

**Note:** One last thing, predictions and accuracies may be subjected to change since I implemented data shuffling on both types of dataframes. The main thing is to see if our models are able to correctly classify no fraud and fraud transactions.