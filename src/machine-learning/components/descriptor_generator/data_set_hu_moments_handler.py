import csv
import glob
import numpy as np
from utils.hu_moments_generation import hu_moments_of_file
from utils.path_helper import project_root


class DatasetHuMomentsHandler:
    def __init__(
        self,
        shape_path: str,
        output_path: str,
    ):
        self.shape_path = shape_path
        self.output_path = output_path

    def write_hu_moments(self, label, writer):
        root = project_root()
        folder = root / self.shape_path / label

        print("Buscando archivos en:", folder)

        files = list(folder.glob("*"))

        print("files =", files)

        for file in files:
            moments = hu_moments_of_file(file)
            row = np.append(moments.ravel(), label)
            writer.writerow(row)


    def generate_hu_moments_file(self):
        root = project_root()
        generate_path = root / self.output_path

        print("root =", root)
        print("full shape path =", root / self.shape_path)
        print("generate_path: ", generate_path)
        with open(generate_path, 'w', newline='') as file:  # generate a new file (W=Write)
            writer = csv.writer(file)
            self.write_hu_moments("diamond", writer)
            self.write_hu_moments("trebol", writer)
            self.write_hu_moments("heart", writer)

        # for label in self.labels:
        #     self.write_hu_moments(label, writer)





