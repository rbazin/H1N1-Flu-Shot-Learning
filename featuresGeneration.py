#%%
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from autofeat import AutoFeatClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

#%%
def categorical_imputing(data, categorical_list):
    # Imputing
    imputer_categorical = SimpleImputer(strategy='constant', fill_value='missing', missing_values=np.nan)
    data_categorical = data.loc[:, categorical_list]
    data_categorical = imputer_categorical.fit_transform(data_categorical)
    data_categorical_imputed = pd.DataFrame(data_categorical, columns=categorical_list)
    return data_categorical_imputed

def numerical_impute(data, numerical_list):
    imputer_numerical = SimpleImputer(strategy='constant', fill_value=-1, missing_values=np.nan)
    data_numerical = data.loc[:, numerical_list]
    data_numerical_imputed = imputer_numerical.fit_transform(data_numerical)
    data_numerical_imputed = pd.DataFrame(data_numerical_imputed, columns=numerical_list)
    return data_numerical_imputed


def train_autofeat_model(data, labels, numerical_list, categorical_list, featen_steps=3, save_file="autofeatModel.pkl"):

    data = data.drop("respondent_id", axis=1)
    labels = labels.loc[:, ['h1n1_vaccine']].to_numpy() # Ne peut malheureusement pas classer deux labels en même temps : faire deux algorithmes

    # Imputing data
    data_numerical_imputed = numerical_impute(data, numerical_list)
    data_categorical_imputed = categorical_imputing(data, categorical_list)

    #Scaling numerical data
    scaler = StandardScaler()
    data_numerical_scaled = scaler.fit_transform(data_numerical_imputed.to_numpy())
    data_numerical_scaled = pd.DataFrame(data_numerical_scaled, columns=numerical_list)

    # Merging data
    data_imputed_scaled = pd.merge(data_numerical_scaled, data_categorical_imputed, left_index=True, right_index=True)

    # Autofeat classifier
    model = AutoFeatClassifier(categorical_cols=categorical_list, feateng_steps=featen_steps, verbose=1)
    model.fit(data_imputed_scaled, labels)

    # On sauvegarde le modèle permettant de générer les nouvelles features
    joblib.dump(model, save_file)


def generate_test_features(data, numerical_list, categorical_list, save_file="autofeatModel.pkl"):
    # On applique le pipeline de transformation des donnée
    respondent_id = data.loc[:, ['respondent_id']]
    data = data.drop("respondent_id", axis=1)

    # Imputing data
    data_numerical_imputed = numerical_impute(data, numerical_list)
    data_categorical_imputed = categorical_imputing(data, categorical_list)

    #Scaling numerical data
    scaler = StandardScaler()
    data_numerical_scaled = scaler.fit_transform(data_numerical_imputed.to_numpy())
    data_numerical_scaled = pd.DataFrame(data_numerical_scaled, columns=numerical_list)

    # Merging data
    data_imputed_scaled = pd.merge(data_numerical_scaled, data_categorical_imputed, left_index=True, right_index=True)

    # On charge le modèle
    model = joblib.load(save_file)

    # On génère les features du fichier test
    new_features = model.transform(data_imputed_scaled) # Est ce qu'il faut donner un nom d'attribut aux nouvelles features ?

    # On ajoute la colonne respondent Id
    new_features = pd.merge(respondent_id, new_features, left_index=True, right_index=True)

    # On sauvegarde les nouvelles données dans un fichier csv
    new_features.to_csv("new_features_test.csv", sep=",", header=True, index=False)






#%%

# Chemin vers les données
LABELS_TRAINING_PATH = "training_set_labels.csv"
FEATURES_TRAINING_PATH = "training_set_features.csv"

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
categorical_list = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa','employment_industry', 'employment_occupation']
categorical_list_one_hot = ['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']
categorical_list_ordinal = [k for k in categorical_list if k not in categorical_list_one_hot]
numerical_list = [k for k in features_list if k not in categorical_list]

#%%

# Entrainnement du modèle autofeat
train_autofeat_model(data, labels, numerical_list, categorical_list)

#%%

# On charge les données de test
TEST_FEATURES_PATH = "test_set_features.csv"
data_test = pd.read_csv(TEST_FEATURES_PATH, sep=",", header=0)

# Génération des nouvelles features à partir du set de test
generate_test_features(data_test, numerical_list, categorical_list)