# FUNCTIONS FOR EXPLORATORY DATA ANALYSIS HOME CREDIT PROJECT

import pandas as pd
import numpy as np
import pymysql as pymysql
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, ClassPredictionError
from yellowbrick.model_selection import LearningCurve, FeatureImportances

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def cat_summary(data, categorical_cols, target, number_of_classes=25):
    '''

    :param data:
    :param categorical_cols:
    :param target:
    :param number_of_classes:
    :return:
    '''
    var_count = 0  # reporting how many categorical variables are there?
    vars_more_classes = []  # save the variables that have classes more than a number that we determined
    for var in categorical_cols:
        if len(data[var].value_counts()) <= number_of_classes:  # select according to number of classes
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


# Function to plot histograms for numerical variables
def hist_for_nums(data, numeric_cols):
    '''

    :param data:
    :param numeric_cols:
    :return:
    '''
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


# Find correlations for numeric variables
def find_correlation(dataframe, num_cols, corr_limit=0.60):
    '''

    :param dataframe:
    :param num_cols:
    :param corr_limit:
    :return:
    '''
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "TARGET":
            pass
        else:
            correlation = dataframe[[col, "TARGET"]].corr().loc[col, "TARGET"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


# Evaluate each model in turn by looking at train and test errors and scores
def evaluate_classification_model_holdout(models, X_train, X_test, y_train, y_test):

    # Define lists to track names and results for models
    names = []
    train_accuracy_results = []
    test_accuracy_results = []
    test_f1_scores = []
    test_roc_auc_scores = []
    test_precision_scores = []
    test_recall_scores = []
    supports = []

    print('################ Accuracy scores for test set for the models: ################\n')
    for name, model in models:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy_result = accuracy_score(y_train, y_train_pred)
        test_accuracy_result = accuracy_score(y_test, y_test_pred)
        train_accuracy_results.append(train_accuracy_result)
        test_accuracy_results.append(test_accuracy_result)
        test_f1_score = f1_score(y_test, y_test_pred)
        test_f1_scores.append(test_f1_score)
        test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
        test_roc_auc_scores.append(test_roc_auc_score)
        test_precision_score = precision_score(y_test, y_test_pred)
        test_precision_scores.append(test_precision_score)
        test_recall_score = recall_score(y_test, y_test_pred)
        test_recall_scores.append(test_recall_score)

        names.append(name)
        msg = "%s: Accuracy: %f, ROCAUCScore: %f, F1 Score: %f" % (name, test_accuracy_result, test_roc_auc_score, test_f1_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': names,
                                'accuracy_train': train_accuracy_results,
                                'accuracy_test': test_accuracy_results,
                                'roc_auc_score': test_roc_auc_scores,
                                'f1_score_test': test_f1_scores,
                                'precision_test': test_precision_scores,
                                'recall_test': test_recall_scores})
    data_result.set_index('models')
    print(data_result)

    # Plot comparison table for accuracy scores
    plt.figure(figsize=(15, 12))
    sns.barplot(x='accuracy_test', y='models', data=data_result.sort_values(by="accuracy_test", ascending=False), color="r")
    plt.xlabel('Accuracy Scores')
    plt.ylabel('Models')
    plt.title('Accuracy Scores For Test Set')
    plt.show()

    # Plot comparison table for roc_auc_scores
    plt.figure(figsize=(15, 12))
    sns.barplot(x='roc_auc_score', y='models', data=data_result.sort_values(by="roc_auc_score", ascending=False), color="salmon")
    plt.xlabel('ROC AUC Scores')
    plt.ylabel('Models')
    plt.title('ROC AUC Scores For Test Set')
    plt.show()


