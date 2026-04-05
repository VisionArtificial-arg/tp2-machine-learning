import csv
import numpy as np
from sklearn import tree
from joblib import dump

from utils.label_converters import label_to_int


class DecisionTreeTrainer:
    def __init__(self, dataset_path: str, model_output_path: str):
        self.dataset_path = dataset_path
        self.model_output_path = model_output_path

    def load_dataset(self):
        X = []
        Y = []

        with open(self.dataset_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                label = row.pop()          # último valor
                values = [float(x) for x in row]
                X.append(values)
                Y.append(label_to_int(label))

        return np.array(X), np.array(Y)

    # Training
    def train(self):
        X, Y = self.load_dataset()

        model = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            random_state=42
        )

        model.fit(X, Y)
        return model

    def save(self, model):
        dump(model, self.model_output_path)
        print(f"Modelo guardado en {self.model_output_path}")