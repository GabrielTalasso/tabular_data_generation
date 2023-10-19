from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class MLE():

    def __init__(self, 
                 synthetic_dataframe,
                 real_dataframe,
                 target_column = 'target'):
        
        self.synthetic_dataframe = synthetic_dataframe
        self.real_dataframe = real_dataframe
        self.target_column = target_column

    def eval(self, metric):

        train = self.synthetic_dataframe.drop(self.target_column, axis = 1)
        train_target = self.synthetic_dataframe[self.target_column]

        test = self.real_dataframe.drop(self.target_column, axis = 1)
        test_target = self.real_dataframe[self.target_column]

        ABC = AdaBoostClassifier(random_state=1)
        ABC.fit(train, train_target)
        abc_pred = ABC.predict(test)
        acc_abc = accuracy_score(abc_pred, test_target)


        LRC = LogisticRegression(random_state=1)
        LRC.fit(train, train_target)
        lrc_pred = LRC.predict(test)
        acc_lrc = accuracy_score(lrc_pred, test_target)


        MLPC = MLPClassifier(hidden_layer_sizes=[10,10], random_state=1)
        mlpc_pred = MLPC.predict(test)
        acc_mlpc = accuracy_score(mlpc_pred, test_target)


        DTC = DecisionTreeClassifier(random_state=1)
        DTC.fit(train, train_target)
        dtc_pred = DTC.predict(test)
        acc_dtc = accuracy_score(dtc_pred, test_target)
