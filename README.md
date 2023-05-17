# Problem definition
The problem we aim to solve in this project is to predict the body level of a human based on 16 different attributes related to their physical, genetic, and habitual conditions. The body level can be categorized into 4 different classes, and it is imbalanced, meaning that some classes have significantly more samples than others. Our goal is to develop machine learning models that can accurately predict the body level while also accounting for the class imbalance.

# Motivation
The ability to accurately classify human body levels can have significant implications in healthcare and medicine. For example, it can help doctors and medical professionals identify potential health risks and develop personalized treatment plans. Additionally, accurate body level classification can aid in the development of new drugs and therapies by allowing researchers to more effectively study the effects of treatments on different body levels. By developing machine learning models that can accurately predict body levels, we can improve our understanding of human physiology and ultimately improve healthcare outcomes for individuals and society as a whole.

# Evaluation metrics: F1-Score

# Dataset: 
    https://drive.google.com/drive/folders/1VyxBvMNaPISko9SrGh7laECsQIM6DiWY?fbclid=IwAR0_Xv8pBf7jkfTgJGeiIkFPS65Co8JGbltrrUJQzVB3p0S-KZuLc-0iDa0

# Preprocessing & Feature Engineering

![e22e04d9-8adc-4c62-84ae-5ebe093369fd](https://github.com/ahmedayman1420/Human-Body-Level-Classification/assets/76254195/8e3dcece-79fd-4cf5-8fdb-44f693531824)
![f6a8309e-9280-4af3-b585-d77acb6912f1](https://github.com/ahmedayman1420/Human-Body-Level-Classification/assets/76254195/bcd7a933-b082-48d8-8110-5d2efd315125)
![6fce6124-6c0e-405a-af0e-25c7ba490871](https://github.com/ahmedayman1420/Human-Body-Level-Classification/assets/76254195/fc34791d-f861-47be-bd2c-430142e4c7eb)

# Models & Classifiers

## Random forest
We splitted the data into 80% training and 20% test sets. We tried random forest with and without the BMI feature and we noticed that: If we include the BMI feature the simple random forest model with default hyperparameters achieves a weighted F1-score of 100% on the test set, However, if we exclude the BMI feature the model achieves a weighted F1-score of 94.23% on the test set. Since the random forest model using BMI with default hyperparameters already achieves a very high accuracy and F1-score, we tried to tune the hyperparameters around the default values using grid search and cross-validation, but the hyperparameter tuning had insignificant effect on the accuracy and F1-score. However, if we fine-tuned the model without the BMI feature using grid search and cross-validation, the weighted F1-score on the test set increased from 94.23% to 94.63% but still lower than the model with BMI feature.

## Logistic regression
We splitted the data into 80% training and 20% test sets. We tried logistic regression with and without the BMI feature and we noticed that: If we include the BMI feature the simple logistic regression model with default hyperparameters achieves a weighted F1-score of 95.14% on the test set, However, if we exclude the BMI feature it achieves a weighted F1-score of 93.23% on the test set. Since the main hyperparameters that affect the performance of the logistic regression model are the solver, penalty and C, we used grid search to try all combinations of them and find the best values for those hyperparameters. The weighted F1-score of the model with the BMI feature increased from 95.14% to 98.32% after using grid search to find the best hyperparameters. The weighted F1-score of the model without the BMI feature increased from 93.23% to 98.33% after using grid search, achieving the same weighted F1-score as the model with the BMI feature.

## Perceptron
After we’ve seen before at the preprocessing stage, BMI feature could make a standing difference with the model accuracy, so we tried many combinations of features: (All features without BMI, All features including BMI, BMI and Height, BMI and Weight)

## Support vector machine
We splitted the data into 80% training and 20% test sets. We tried SVM with and without the BMI feature and we noticed that, if we include the BMI feature the simple SVM model with default hyperparameters achieves a weighted F1-score of 96.51% on the test set, However, if we exclude the BMI feature it achieves a weighted F1-score of 96.14% on the test set. Since the main hyperparameters that affect the performance of the SVM model are the C, kernel and gamma, we used grid search to try given combinations of them and find the best values for those hyperparameters. Best estimator of SVM out SVC(C=5, gamma=1, kernel='linear').

# Experimental results

## Random Forest - With BMI
Weighted F1-score | Accuracy
------------ | -------------
    100%     |     100%
    
## Random Forest - Without BMI
Weighted F1-score | Accuracy
------------ | -------------
    94.63%     |     94.59%

## Logistic regression - With BMI
Weighted F1-score | Accuracy
------------ | -------------
    98.32%   |     98.31%
 
## Logistic regression - Without BMI
Weighted F1-score | Accuracy
------------ | -------------
    98.33%   |    98.31%
 
 ## SVM - With BMI
Weighted F1-score | Accuracy
------------ | -------------
    96.91%   |     98.31%

## SVM - Without BMI
Weighted F1-score | Accuracy
------------ | -------------
    96.21%   |     98.12%


















