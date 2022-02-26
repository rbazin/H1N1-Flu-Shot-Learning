import os
import joblib
import numpy as np


def chain_stack_predict_proba(data, path, chain=1):
    """ Make a proba prediction based on all the models in paths """

    if (chain == 1):
        # 1. We want to get all the models in paths but the stacking LR and predict proba of data
        model_files = os.listdir(os.path.join(path, "h1n1"))

        model = joblib.load(model_files[0])
        first_pred = [model.predict_proba(data)[:, 1]]

        for i in range(1, len(model_files)):
            model = joblib.load(model_files[i])
            first_pred = np.c_[first_pred, model.predict_proba(data)[:, 1]]

        # 2. We use the predictions of the models with the stacking LR to predict the proba of h1n1
        stacking_h1n1 = joblib.load(os.path.join(
            path, "stacking", "stacking_h1n1.save"))
        stacking_h1n1_pred = stacking_h1n1.predict_proba(first_pred)[:, 1]

        # 3. We add this prediction to the dataset
        data = np.c_[data, stacking_h1n1_pred]

        # 4. We want to get all the models in paths but the stacking LR and predict proba of data + previous predictions
        model_files = os.listdir(os.path.join(path, "seasonal"))
        model = joblib.load(model_files[0])
        second_pred = [model.predict_proba(data)[:, 1]]

        for i in range(1, len(model_files)):
            model = joblib.load(model_files[i])
            second_pred = np.c_[second_pred, model.predict_proba(data)[:, 1]]

        # 5. We use the predictions of the models with the stacking LR to predict the proba of seasonal
        stacking_seasonal = joblib.load(os.path.join(
            path, "stacking", "stacking_seasonal.save"))
        stacking_seasonal_pred = stacking_seasonal.predict_proba(second_pred)[
            :, 1]

        pred = np.c_[stacking_h1n1_pred, stacking_seasonal_pred]

        return pred

    elif (chain == 2):
        # 1. We want to get all the models in paths but the stacking LR and predict proba of data
        model_files = os.listdir(os.path.join(path, "seasonal"))

        model = joblib.load(model_files[0])
        first_pred = [model.predict_proba(data)[:, 1]]

        for i in range(1, len(model_files)):
            model = joblib.load(model_files[i])
            first_pred = np.c_[first_pred, model.predict_proba(data)[:, 1]]

        # 2. We use the predictions of the models with the stacking LR to predict the proba of h1n1
        stacking_seasonal = joblib.load(os.path.join(
            path, "stacking", "stacking_seasonal.save"))
        stacking_seasonal_pred = stacking_seasonal.predict_proba(first_pred)[
            :, 1]

        # 3. We add this prediction to the dataset
        data = np.c_[data, stacking_seasonal_pred]

        # 4. We want to get all the models in paths but the stacking LR and predict proba of data + previous predictions
        model_files = os.listdir(os.path.join(path, "h1n1"))
        model = joblib.load(model_files[0])
        second_pred = [model.predict_proba(data)[:, 1]]

        for i in range(1, len(model_files)):
            model = joblib.load(model_files[i])
            second_pred = np.c_[second_pred, model.predict_proba(data)[:, 1]]

        # 5. We use the predictions of the models with the stacking LR to predict the proba of seasonal
        stacking_h1n1 = joblib.load(os.path.join(
            path, "stacking", "stacking_h1n1.save"))
        stacking_h1n1_pred = stacking_h1n1.predict_proba(second_pred)[:, 1]

        pred = np.c_[stacking_h1n1_pred, stacking_h1n1_pred]

        return pred
