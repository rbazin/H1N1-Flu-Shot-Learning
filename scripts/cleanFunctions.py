import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

""" Fonctions de traintement des donnees """


def numerical_impute(data, numerical_list):
    imputer_numerical = SimpleImputer(
        strategy='constant', fill_value=-1, missing_values=np.nan)
    data_numerical = data.loc[:, numerical_list]
    data_numerical_imputed = imputer_numerical.fit_transform(data_numerical)
    data_numerical_imputed = pd.DataFrame(
        data_numerical_imputed, columns=numerical_list)
    return data_numerical_imputed


def categorical_imputing(data, categorical_list):
    # Imputing
    imputer_categorical = SimpleImputer(
        strategy='constant', fill_value='missing', missing_values=np.nan)
    data_categorical = data.loc[:, categorical_list]
    data_categorical = imputer_categorical.fit_transform(data_categorical)
    data_categorical_imputed = pd.DataFrame(
        data_categorical, columns=categorical_list)
    return data_categorical_imputed


def categorical_impute_one_hot(data, categorical_list):
    # Imputing
    data_categorical_imputed = categorical_imputing(data, categorical_list)

    # One hot encoding
    data_one_hot = pd.get_dummies(data_categorical_imputed)

    return data_one_hot


def data_clean(data, numerical_list, categorical_list):
    # Changer les listes de features et les fonctions correspondantes
    data_categorical_encoded = categorical_impute_one_hot(
        data, categorical_list)
    data_numerical_imputed = numerical_impute(data, numerical_list)
    data_imputed_encoded = pd.merge(
        data_numerical_imputed, data_categorical_encoded, left_index=True, right_index=True)

    return data_imputed_encoded
