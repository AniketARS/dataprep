import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class Preprocessor(object):
    __instance = None
    _encoder_map = dict()
    _scalar_map = dict()
    _algos = []
    _type_map = dict()

    @staticmethod
    def instance():
        if Preprocessor.__instance is None:
            Preprocessor()
        return Preprocessor.__instance

    def __init__(self):
        if Preprocessor.__instance is None:
            Preprocessor.__instance = self
            self._init_algos()
            self._init_type_map()
        else:
            raise RuntimeError('Class is Singleton, Please call instance() instead')

    def generate_stats(self, combined: list, columns: list):
        temp_df: pd.DataFrame = combined[0].copy()
        for df in combined[1:]:
            temp_df = temp_df.append(df, ignore_index=True)
        if len(columns) > 0:
            for col in temp_df.columns:
                if col not in columns:
                    temp_df = temp_df.drop(col, axis=1)
        temp_df.describe().to_csv('./dataset/stats.csv')

    def encode(self, df: pd.DataFrame, columns: list, en_type: str = 'label', drop_one: bool = False):
        new_df = df.copy()
        if en_type == 'label':
            for col in columns:
                self._encoder_map[col] = self._type_map['encoder'][en_type]()
                new_df[col] = self._encoder_map[col].fit_transform(new_df[col])
        elif en_type == 'onehot':
            new_df = pd.get_dummies(new_df, columns=columns, prefix=columns)
            if drop_one:
                cols = new_df.columns
                r = re.compile(columns[0] + '_*')
                to_del = list(filter(r.match, cols))[0]
                new_df = new_df.drop(to_del, axis=1)
        return new_df

    def scale(self, df: pd.DataFrame, columns: list, scale_type: str):
        new_df = df.copy()
        for col in columns:
            self._scalar_map[col] = self._type_map['scalar'][scale_type]()
            arr = new_df[col].to_numpy().reshape(-1, 1)
            arr = self._scalar_map[col].fit_transform(arr).reshape(-1)
            new_df[col] = arr
        return new_df

    def test(self, X, y):
        algo_name = []
        algo_acc = []
        for algo in self._algos:
            algo.fit(X, y)
            algo.score(X, y)
            acc = round(algo.score(X, y) * 100, 2)
            algo_name.append(str(type(algo)).split('.')[-1][:-2])
            algo_acc.append(acc)
        return pd.DataFrame({'Model': algo_name, 'Score': algo_acc})

    def _init_algos(self):
        if len(self._algos) == 0:
            self._algos = [
                SGDClassifier(max_iter=5, tol=None),
                RandomForestClassifier(n_estimators=100),
                LogisticRegression(),
                KNeighborsClassifier(n_neighbors=3),
                GaussianNB(),
                Perceptron(max_iter=5),
                LinearSVC(),
                DecisionTreeClassifier(),
            ]

    def _init_type_map(self):
        self._type_map = {
            'encoder': {
                'label': self._label_encoder
            },
            'scalar': {
                'standard': self._standard_scalar
            }
        }

    def _label_encoder(self):
        return LabelEncoder()

    def _standard_scalar(self):
        return StandardScaler()
