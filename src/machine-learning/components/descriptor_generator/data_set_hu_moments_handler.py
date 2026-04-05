import csv
import glob
import numpy as np
from utils.hu_moments_generation import hu_moments_of_file


class DatasetHuMomentsHandler:
    def __init__(
        self,
        # labels: list[str],
        shape_path: str,
        output_path: str,
        # grayscale,
        # threshold,
        # erosion,
    ):
        # self.labels = labels
        self.shape_path = shape_path
        self.output_path = output_path
        # self.grayscale = grayscale
        # self.threshold = threshold
        # self.erosion = erosion

    def write_hu_moments(self, label, writer):
        folder = f"{self.shape_path}/{label}/*"
        files = glob.glob(folder)

        for file in files:
            moments = hu_moments_of_file(file)
            row = np.append(moments.ravel(), label)
            writer.writerow(row)


    def generate_hu_moments_file(self):
        with open('../generated_files/shapes-hu-moments.csv', 'w', newline='') as file:  # generate a new file (W=Write)
            writer = csv.writer(file)
            self.write_hu_moments("5-point-star", writer)
            self.write_hu_moments("rectangle", writer)
            self.write_hu_moments("triangle", writer)

        # for label in self.labels:
        #     self.write_hu_moments(label, writer)





