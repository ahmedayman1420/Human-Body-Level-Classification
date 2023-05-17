from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from mlxtend.evaluate import bias_variance_decomp
import numpy as np

def apply_crossvalidation(model, X_train, Y_train, k=10):

    # create a k-fold cross-validation iterator
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # perform k-fold cross-validation and compute accuracy
    scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
    # print the average accuracy score and its standard deviation
    print('Accuracy: {} +/- {}'.format(scores.mean(), scores.std()))

    # perform k-fold cross-validation and compute F1-score
    scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='f1_weighted')
    # print the average F1-score and its standard deviation
    print('F1-score: {} +/- {}'.format(scores.mean(), scores.std()))

    # # perform k-fold cross-validation and compute AUC
    # scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='roc_auc')
    # # print the average AUC score and its standard deviation
    # print('AUC: {} +/- {}'.format(scores.mean(), scores.std()))

def Evaluate(model, X_test, Y_test):
    
    # predict the class labels for the test set
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(Y_test, y_pred)

    # calculate the precision
    precision = precision_score(Y_test, y_pred, average='weighted')

    # calculate the recall
    recall = recall_score(Y_test, y_pred, average='weighted')

    # calculate the F1 score
    f1 = f1_score(Y_test, y_pred, average='weighted')

    # calculate the confusion matrix
    cm = confusion_matrix(Y_test, y_pred)

    # print the results
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Confusion matrix:\n', cm)


def draw_learning_curve(model, x_train, y_train):
    # Define the number of training samples at each iteration
    train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate the mean and standard deviation of the training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()

    # Plot the mean training and test scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Plot the shaded area indicating the variance (± one standard deviation)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Add a legend
    plt.legend(loc="best")

    # Show the plot
    plt.show()


def draw_bias_variance_decomp(model, x_train, y_train, x_test, y_test):
    # Calculate bias and variance
    mse, bias, variance = bias_variance_decomp(model, x_train.values, y_train.values, x_test.values, y_test.values, loss='mse', num_rounds=200, random_seed=42)

    print("Mean Squared Error:", mse)
    print("Bias:", bias)
    print("Variance:", variance)