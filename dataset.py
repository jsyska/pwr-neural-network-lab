import os

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler


class MyDataClass:
    def __init__(self):
        self.y_test = None
        self.X_test = None
        self.y_train = None
        self.X_train = None

    def load_and_preprocess_heart_disease(self):
        self.load_data_heart_disease()
        self.preprocess_data_heart_disease()

    def load_data_heart_disease(self):
        self.X_train = pd.read_csv(f'data/processed/X_train_heart_disease.csv')
        self.y_train = pd.read_csv(f'data/processed/y_train_heart_disease.csv')
        self.X_test = pd.read_csv(f'data/processed/X_test_heart_disease.csv')
        self.y_test = pd.read_csv(f'data/processed/y_test_heart_disease.csv')

    def preprocess_data_heart_disease(self):
        for column in self.X_train.columns:
            if self.X_train[column].isna().any():
                mean_value = self.X_train[column].mean()
                self.X_train[column].fillna(mean_value, inplace=True)


        for column in self.X_test.columns:
            if self.X_test[column].isna().any():
                mean_value = self.X_test[column].mean()
                self.X_test[column].fillna(mean_value, inplace=True)


        self.X_train = self.X_train.to_numpy()
        self.y_train = self.y_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.y_test = self.y_test.to_numpy()

        if len(self.y_train.shape) > 1 and self.y_train.shape[1] == 1:
            self.y_train = self.y_train.flatten()
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] == 1:
            self.y_test = self.y_test.flatten()

        scaler = StandardScaler()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def load_and_preprocess_affnist(self, data_dir):
        self.load_data_affnist(data_dir)
        self.preprocess_data_affnist()

    def load_data_affnist(self, data_dir):
        train_data = self.load_matlab(os.path.join(data_dir, 'affnist_training.mat'))
        self.X_train, self.y_train = train_data

        test_data = self.load_matlab(os.path.join(data_dir, 'affnist_test.mat'))
        self.X_test, self.y_test = test_data

    def preprocess_data_affnist(self):
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        self.X_train = self.X_train.reshape(-1, 1600)
        self.X_test = self.X_test.reshape(-1, 1600)

        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()

    @staticmethod
    def load_matlab(file):
        def _check_keys(matlab_data):
            for key in matlab_data:
                if isinstance(matlab_data[key], scipy.io.matlab.mio5_params.mat_struct):
                    matlab_data[key] = _to_dict(matlab_data[key])
            return matlab_data

        def _to_dict(matlab_object):
            output_data = {}
            for name in matlab_object._fieldnames:
                elem = matlab_object.__dict__[name]
                if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                    output_data[name] = _to_dict(elem)
                else:
                    output_data[name] = elem
            return output_data

        data = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
        data = _check_keys(data)

        x = data["affNISTdata"]["image"].transpose() / 255
        y = data["affNISTdata"]["label_int"]

        return x, y

    def load_and_preprocess_forest_fires(self):
        self.load_data_forest_fires()
        self.preprocess_data_forest_fires()

    def load_data_forest_fires(self):
        self.X_train = pd.read_csv('data/processed/X_train_forest_fires.csv')
        self.y_train = pd.read_csv('data/processed/y_train_forest_fires.csv')
        self.X_test = pd.read_csv('data/processed/X_test_forest_fires.csv')
        self.y_test = pd.read_csv('data/processed/y_test_forest_fires.csv')

    def preprocess_data_forest_fires(self):
        self.X_train = self.X_train.to_numpy()
        self.y_train = self.y_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.y_test = self.y_test.to_numpy()


        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)