import os

import cv2
import numpy as np
import glob
from joblib import load

from utils.hu_moments_generation import hu_moments_of_file, hu_moments_of_frame, contour_and_hu_moments_of_frame, contour_hu_pairs_of_frame
from utils.label_converters import int_to_label
from utils.path_helper import project_root


class InferenceRunner:
    def __init__(self, model_path, testing_images_path):
        root = project_root()
        model_path = os.path.join(root, model_path)
        self.model = load(model_path)
        self.testing_images_path = testing_images_path

    def predict_from_hu(self, hu):
        sample = np.array([hu.flatten()], dtype=np.float32)
        prediction = self.model.predict(sample)[0]
        return int_to_label(prediction)

    def predict_from_frame(self, frame):
        hu = hu_moments_of_frame(frame)
        if hu is None:
            return None  # si no encuentra contornos
        return self.predict_from_hu(hu)

    def predict_with_contour_from_frame(self, frame):
        contour, hu = contour_and_hu_moments_of_frame(frame)
        if hu is None:
            return None, None
        return self.predict_from_hu(hu), contour

    def predict_many_from_frame(self, frame, min_area=500, max_area_ratio=0.98, ignore_border=False):
        predictions = []
        for contour, hu in contour_hu_pairs_of_frame(
            frame,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            ignore_border=ignore_border,
        ):
            label = self.predict_from_hu(hu)
            predictions.append((label, contour))
        return predictions



    def run(self):
        files = glob.glob(f"{self.testing_images_path}/*")

        for f in files:
            hu = hu_moments_of_file(f) # generates descriptors from image

            # sklearn expect shape = (1, 7)
            sample = np.array([hu.flatten()], dtype=np.float32)

            prediction = self.model.predict(sample)[0] # prediction

            # read image + draw prediction
            image = cv2.imread(f)
            label = int_to_label(prediction)

            annotated = cv2.putText(
                image,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Result", annotated)
            print(f"Image: {f} -> Prediction: {label}")
            cv2.waitKey(0)

        cv2.destroyAllWindows()


