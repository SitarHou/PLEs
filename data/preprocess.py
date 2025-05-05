import warnings
import pandas as pd
import os
import json
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def outlier_detect_IQR(data, col, threshold=3):

    """
    outlier detection by Interquartile Ranges Rule, also known as Tukey's test.
    calculate the IQR ( 75th quantile - 25th quantile)
    and the 25th 75th quantile.
    Any value beyond:
        upper bound = 75th quantile + （IQR * threshold）
        lower bound = 25th quantile - （IQR * threshold）
    are regarded as outliers. Default threshold is 3.
    """

    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    outlier_counts = outlier_index.value_counts()
    num_outliers = outlier_counts.get(1, 0)
    total_count = len(outlier_index)
    #print('Num of outlier detected:', num_outliers)
    #print('Proportion of outlier detected', num_outliers / total_count if total_count > 0 else 0)
    return outlier_index, para

def outlier_detect_arbitrary(data, col, upper_fence, lower_fence):
    """
    identify outliers based on arbitrary boundaries passed to the function.
    """
    para = (upper_fence, lower_fence)
    tmp = pd.concat([data[col] > upper_fence, data[col] < lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    #print(outlier_index.value_counts())
    #print('Num of outlier detected:', outlier_index.value_counts()[1])
    #print('Proportion of outlier detected', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index, para


def impute_outlier_with_avg(data, col, outlier_index, strategy='mean'):
    """
    impute outlier with mean/median/most frequent values of that variable.
    """
    data_copy = data.copy(deep=True)
    if strategy == 'mean':
        data_copy.loc[outlier_index, col] = data_copy[col].mean()
    elif strategy == 'median':
        data_copy.loc[outlier_index, col] = data_copy[col].median()
    elif strategy == 'mode':
        data_copy.loc[outlier_index, col] = data_copy[col].mode()[0]

    return data_copy

def impute_standard(train, test):
    """
    imputation by Iterative Imputer
    standardization by StandardScaler()
    return: preprocessed datasets -->train, test
    """
    X_train = train.copy()
    X_test = test.copy()

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', 'config', 'categorical_dict.json')
    with open(config_path, 'r') as f:
        cat_dict = json.load(f)
    cat_features = [key for key, value in cat_dict.items() if value and key in X_train.columns]
    con_features = set(X_train.columns[1:]) - set(cat_features)

    con_train = X_train[list(con_features)].copy()
    con_test = X_test[list(con_features)].copy()
    con_train.replace([np.inf, -np.inf], 0, inplace=True)
    con_test.replace([np.inf, -np.inf], 0, inplace=True)

    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputer.fit(con_train)
    # imputer_path = os.path.join(current_dir, '..', 'config', 'iterative_imputer_model.pkl')
    # imputer = joblib.load(imputer_path)
    # joblib.dump(imputer, imputer_path)

    # original_columns = imputer.feature_names_in_
    # con_train = con_train[original_columns]
    # con_test = con_test[original_columns]
    train_imputed = imputer.transform(con_train)
    train_imputed_df = pd.DataFrame(train_imputed, columns=list(con_features))
    test_imputed = imputer.transform(con_test)
    test_imputed_df = pd.DataFrame(test_imputed, columns=list(con_features))

    scaler = StandardScaler()
    scaler.fit(train_imputed_df)
    X_train_scaled = scaler.transform(train_imputed_df)
    X_test_scaled = scaler.transform(test_imputed_df)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=list(con_features))
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=list(con_features))

    return X_train_scaled, X_test_scaled