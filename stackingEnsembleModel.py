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

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Fonctions de traitement des donnees


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


# Chemin vers les données et le nom du fichier de sortie
LABELS_TRAINING_PATH = os.path.join("data", "training_set_labels.csv")
FEATURES_TRAINING_PATH = os.path.join("data", "training_set_features.csv")
BEST_PARAMS_PATH = os.path.join("models", "best_params_skmultilearn.save")
VOTE_H1N1_PATH = os.path.join("models", "vote_h1n1_clf.save")
VOTE_SEASONAL_PATH = os.path.join("models", "vote_seasonal_clf.save")

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
    X, Y, test_size=0.2, random_state=1)

# On scale les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


""" Dexuième partie : sklearn stacking ensemble learning """

# La stratégie est d'utiliser la relation entre les labels grâce à une transformation chainée
models = [
    ('lr', LogisticRegression()),
    ('rndf', RandomForestClassifier(n_estimators=300)),
    ('xgb', XGBClassifier()),
    ('ada', AdaBoostClassifier()),
    ('cb', CatBoostClassifier()),
    ('svc', SVC(probability=True, kernel="poly", degree=3)),
    ('gsb', GaussianNB()),
]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Premier maillon de la chaine qui va prédire 'h1n1_vaccine'
vote_clf_h1n1 = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression(),
    cv=cv,
    n_jobs=-1,
)

vote_clf_h1n1.fit(X_train_scaled, Y_train[:, 0])
joblib.dump(vote_clf_h1n1, VOTE_H1N1_PATH)

# On prépare les donées d'entrainnement pour le deuxième maillon de la chaine
clf_h1n1_pred = vote_clf_h1n1.predict_proba(X_train_scaled)[:, 1]
X_train_seasonal_scaled = np.c_[X_train_scaled, clf_h1n1_pred]

# Deuxième maillon de la chaine qui va prédire 'seasonal_vaccine'
vote_clf_seasonal = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression(),
    cv=cv,
    n_jobs=-1,
)

vote_clf_seasonal.fit(X_train_seasonal_scaled, Y_train[:, 1])
joblib.dump(vote_clf_seasonal, VOTE_SEASONAL_PATH)

# Prédictions
clf_h1n1_pred = vote_clf_h1n1.predict_proba(X_test_scaled)[:, 1]
X_test_seasonal_scaled = np.c_[X_test_scaled, clf_h1n1_pred]
clf_seasonal_pred = vote_clf_seasonal.predict_proba(
    X_test_seasonal_scaled)[:, 1]

pred = np.c[clf_h1n1_pred, clf_seasonal_pred]

print('ROC AUC :', roc_auc_score(Y_test, pred))
