import tensorflow.keras as keras
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def numerical_impute(data, numerical_list):
    imputer_numerical = SimpleImputer(strategy='constant', fill_value=-1, missing_values=np.nan)
    data_numerical = data.loc[:, numerical_list]
    data_numerical_imputed = imputer_numerical.fit_transform(data_numerical)
    data_numerical_imputed = pd.DataFrame(data_numerical_imputed, columns=numerical_list)
    return data_numerical_imputed

def categorical_impute_encode(data, categorical_list):
    # Imputing
    imputer_categorical = SimpleImputer(strategy='constant', fill_value='missing', missing_values=np.nan)
    data_categorical = data.loc[:, categorical_list]
    data_categorical = imputer_categorical.fit_transform(data_categorical)
    data_categorical = pd.DataFrame(data_categorical, columns=categorical_list)

    # Ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    data_categorical_encoded = ordinal_encoder.fit_transform(data_categorical)
    data_categorical_encoded = pd.DataFrame(data_categorical_encoded, columns=categorical_list)
    return data_categorical_encoded

def data_clean(data, numerical_list, categorical_list):
    # Changer les listes de features et les fonctions correspondantes
    data = data.drop("respondent_id", axis=1)
    data_categorical_encoded = categorical_impute_encode(data, categorical_list)
    data_numerical_imputed = numerical_impute(data, numerical_list)
    data_imputed_encoded = pd.merge(data_numerical_imputed, data_categorical_encoded, left_index=True, right_index=True)
    return data_imputed_encoded

# Données
TEST_PATH = "test_set_features.csv"
test_data = pd.read_csv(TEST_PATH, sep=",", header=0)

# On charge le meilleur modèle
model = keras.models.load_model("best_model_ever_8_12_2021.h5") # 86 % de réussite à l'entrainnement

# Chargement des listes d'attributs
lists = joblib.load("lists.save")
numerical_list = lists['numerical_list']
categorical_list = lists['categorical_list']

# On applique la transformation de nettoyage aux données de test
respondent_id = test_data.loc[:, ["respondent_id"]]
data_test_imputed = data_clean(test_data, numerical_list, categorical_list)

# On applique la transformation de scaling aux données de test
scaler = joblib.load("scaler.save")
X = data_test_imputed.to_numpy()
X_scaled = scaler.transform(X)

# Prédiction de 'h1n1_vaccine' et 'seasonal_vaccine'
Y_predicted = model.predict(X_scaled)
df_pred = pd.DataFrame(Y_predicted, columns=['h1n1_vaccine', 'seasonal_vaccine'])
df_pred = pd.merge(respondent_id, df_pred, left_index=True, right_index=True)

# Ecriture du fichier csv de sortie
df_pred.to_csv("predictions_test.csv", sep=',', header=True, index=False)