import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cleanFunctions import *
from stackingFunctions import *
import os

# Données
TEST_PATH = "test_set_features.csv"
test_data = pd.read_csv(TEST_PATH, sep=",", header=0)

# On charge un modèle à tester
PATH_CHAIN1 = os.path.join("..", "models", "chain1")

# Chargement des listes d'attributs
lists = joblib.load("lists.save")
numerical_list = lists['numerical_list']
categorical_list = lists['categorical_list']

# On applique la transformation de nettoyage aux données de test
respondent_id = test_data.loc[:, ["respondent_id"]]
data_test_imputed = data_clean(test_data, numerical_list, categorical_list)

# On applique la transformation de scaling aux données de test
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_test_imputed)

# Prédiction de 'h1n1_vaccine' et 'seasonal_vaccine'
Y_predicted = chain_stack_predict_proba(X_scaled, PATH_CHAIN1, chain=1)
df_pred = pd.DataFrame(Y_predicted, columns=[
                       'h1n1_vaccine', 'seasonal_vaccine'])
df_pred = pd.merge(respondent_id, df_pred, left_index=True, right_index=True)

# Ecriture du fichier csv de sortie
df_pred.to_csv("predictions_test.csv", sep=',', header=True, index=False)
