from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Discriminator():

    def __init__(self,
                 synthetic_dataframe,
                 real_dataframe):
        
        self.synthetic_dataframe = synthetic_dataframe
        self.real_dataframe = real_dataframe
        

        
    def preprocessing(self):

        self.synthetic_dataframe = self.synthetic_dataframe.dropna(axis=0)
        self.real_dataframe      =self.real_dataframe.dropna(axis =0)


        #get_dummies of the categorical columns to apply the ML algorithms
        cols = self.synthetic_dataframe.columns
        num_cols = self.synthetic_dataframe._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        self.synthetic_dataframe = pd.get_dummies(self.synthetic_dataframe, columns=cat_cols)

        cols = self.real_dataframe.columns
        num_cols = self.real_dataframe._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        self.real_dataframe = pd.get_dummies(self.real_dataframe, columns=cat_cols)


        self.synthetic_dataframe['synthetic'] = 1
        self.real_dataframe['synthetic'] = 0

    def fit(self):

        data = pd.concat([self.synthetic_dataframe, self.real_dataframe], axis=0)
        X = data.drop('synthetic', axis = 1)
        y = data['synthetic']
        train, test, train_target, test_target = train_test_split(X, y, train_size=0.8)

        ABC = AdaBoostClassifier(random_state=1)
        ABC.fit(train, train_target)
        abc_pred = ABC.predict(test)
        acc_abc = accuracy_score(abc_pred, test_target)


        LRC = LogisticRegression(random_state=1)
        LRC.fit(train, train_target)
        lrc_pred = LRC.predict(test)
        acc_lrc = accuracy_score(lrc_pred, test_target)


        MLPC = MLPClassifier(hidden_layer_sizes=[10,10], random_state=1)
        MLPC.fit(train, train_target)
        mlpc_pred = MLPC.predict(test)
        acc_mlpc = accuracy_score(mlpc_pred, test_target)


        DTC = DecisionTreeClassifier(random_state=1)
        DTC.fit(train, train_target)
        dtc_pred = DTC.predict(test)
        acc_dtc = accuracy_score(dtc_pred, test_target)

        mean_acc = np.mean([acc_abc, acc_lrc, acc_mlpc,  acc_dtc])

        return mean_acc

