import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV


# Fonctions de traintement des donnees

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


def predict_proba_all_models(models, data):

    pred = models[0].predict_proba(data)[:, 1]
    for i in range(1, len(models)):
        pred = np.c_[pred, models[i].predict_proba(data)[:, 1]]

    return pred

# Fonction d'entrainnement


def train_chains_model(models_and_params, data, labels, chain=1):
    h1n1 = labels[:, 0]
    seasonal = labels[:, 1]
    trained_models_h1n1 = []
    trained_models_seasonal = []

    if chain == 1:
        print("Training the first chain : h1n1 -> seasonal")
        # We train the h1n1 part
        print("Training models for h1n1 label")
        for model_and_params in models_and_params:

            name = model_and_params['name']
            print("Training model :", name)
            model = model_and_params['model']
            param_grid = model_and_params['param_grid']
            grid_clf = GridSearchCV(model, param_grid, cv=5,
                                    scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_clf.fit(data, h1n1)
            trained_models_h1n1.append(grid_clf)
            joblib.dump(grid_clf, os.path.join(
                CHAIN1_H1N1_PATH, "{}.save".format(name)))
            print("Model :", name, "trained and saved")

        # We gather the predictions of all the models trained
        pred_h1n1_all_models = predict_proba_all_models(
            trained_models_h1n1, data)

        # The stacking model is a regression but it could be otherwise
        stack_h1n1 = LogisticRegression()

        # We train the stacking model on the predictions of the base models and h1n1
        print("Training stacking model for h1n1")
        stack_h1n1.fit(pred_h1n1_all_models, h1n1)
        joblib.dump(stack_h1n1, os.path.join(
            CHAIN1_H1N1_PATH, "stacking_h1n1.save"))
        print("Model : stacking h1n1 trained and saved")

        # We add the prediction of the stacking model to the data so it's used in the next step of the chain
        pred_h1n1 = stack_h1n1.predict_proba(pred_h1n1_all_models)[:, 1]
        data = np.c_[data, pred_h1n1]

        # We train the seasonal part
        print("Training models for seasonal label")
        for model_and_params in models_and_params:
            name = model_and_params['name']
            print("Training model :", name)
            model = model_and_params['model']
            param_grid = model_and_params['param_grid']
            grid_clf = GridSearchCV(model, param_grid, cv=5,
                                    scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_clf.fit(data, seasonal)
            trained_models_seasonal.append(grid_clf)
            joblib.dump(grid_clf, os.path.join(
                CHAIN1_SEASONAL_PATH, "{}.save".format(name)))
            print("Model :", name, "trained and saved")

        # We gather the predictions of all the models trained
        pred_seasonal_all_models = predict_proba_all_models(
            trained_models_seasonal, data)

        stack_seasonal = LogisticRegression()

        print("Training stacking model for seasonal")
        stack_seasonal.fit(pred_seasonal_all_models, seasonal)
        joblib.dump(stack_seasonal, os.path.join(
            CHAIN1_SEASONAL_PATH, "stacking_seasonal.save"))
        print("Model : stacking seasonal trained and saved")

        pred_seasonal = stack_seasonal.predict_proba(
            pred_seasonal_all_models)[:, 1]

        # Testing the roc_auc_score on training data
        full_pred = np.c_[pred_h1n1, pred_seasonal]
        print("RocAucScore on training data :",
              roc_auc_score(labels, full_pred))
    else:
        print("Argument error")


# Chemin vers les données et le nom du fichier de sortie
LABELS_TRAINING_PATH = os.path.join("data", "training_set_labels.csv")
FEATURES_TRAINING_PATH = os.path.join(
    "data", "training_set_features.csv")

# Regular chain : first predict h1n1 then seasonal
CHAIN1_H1N1_PATH = os.path.join("models", "chain1", "h1n1")
CHAIN1_SEASONAL_PATH = os.path.join("models", "chain1", "seasonal")

# Alt chain : first predict seasonal then h1n1
CHAIN2_H1N1_PATH = os.path.join("models", "chain2", "h1n1")
CHAIN2_SEASONAL_PATH = os.path.join("models", "chain2", "seasonal")


# On charge les données
features = pd.read_csv(FEATURES_TRAINING_PATH, sep=",", header=0)
labels = pd.read_csv(LABELS_TRAINING_PATH, sep=",", header=0)
data = pd.merge(features, labels, on="respondent_id")
respondent_id = data['respondent_id']

# Listes de features complètes
arg_list = list(data.keys())
features_list = arg_list.copy()
features_list.remove("h1n1_vaccine")
features_list.remove("seasonal_vaccine")
features_list.remove("respondent_id")

# Différentes listes de features utiles
labels_list = ['h1n1_vaccine', 'seasonal_vaccine']
categorical_list = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
                    'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']
categorical_list_one_hot = ['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status',
                            'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']
categorical_list_ordinal = [
    k for k in categorical_list if k not in categorical_list_one_hot]
numerical_list = [k for k in features_list if k not in categorical_list]

#
labels.drop("respondent_id", inplace=True, axis=1)
Y = labels.to_numpy()
X = data_clean(data, numerical_list, categorical_list).to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)

# On scale les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


""" Dexuième partie : chain and stacking ensemble learning """

models_and_params = [
    {
        'name': 'lr',
        'model': LogisticRegression(),
        'param_grid': {
            'C': [1.0, 0.1, 10.0],
        }
    },
    {
        'name': 'rndf',
        'model': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [200, 300, 500],
            'max_features': [5, 10, 20],
        }
    },
    {
        'name': 'svc',
        'model': SVC(),
        'param_grid': {
            'kernel': ['linear', 'poly', 'rbf'],
            'probability': [True],
            'C': [1, 0.1,  0.01],
        }
    },
    {
        'name': 'xgb',
        'model': XGBClassifier(),
        'param_grid': {
            'eta': [0.3, 1, 0.01],
            'max_depth': [3, 7, 10],
        }
    },
    {
        'name': 'ada',
        'model': AdaBoostClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.01, 1],
        },
    },
    {
        'name': 'cb',
        'model': CatBoostClassifier(),
        'param_grid': {
            'n_estimators': [100, 300, 1000],
            'eta': [1, 0.1, 0.01],
            'max_depth': [2, 5, 10],
        }
    },
    {
        'name': 'gsb',
        'model': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-9, 1e-10, 1e-10],
        }

    }
]

train_chains_model(models_and_params, X_train_scaled, Y_train, chain=1)
