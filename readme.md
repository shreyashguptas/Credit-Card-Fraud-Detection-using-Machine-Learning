# Credit Card Fraud Detection

This project focuses on detecting credit card fraud using machine learning techniques. The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

## Dataset

The dataset can be found on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

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

Based on the analysis and experiments conducted in this project, the following conclusions can be drawn:

- Handling imbalanced datasets is crucial for building effective fraud detection models. Techniques like undersampling and oversampling (SMOTE) can help mitigate the class imbalance problem.

- Neural networks, particularly the oversampled model, demonstrate promising results in detecting fraudulent transactions.

- Outlier removal can potentially improve the accuracy of the fraud detection model. By removing outliers using the Z-score method, the model's performance on the test set is enhanced.

It's important to note that the predictions and accuracies may vary due to data shuffling implemented in the code. The main objective is to assess the models' ability to correctly classify fraudulent and non-fraudulent transactions.

For more details on the code and analysis, please refer to the `fraud-detection.ipynb` notebook.