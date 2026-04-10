import cv2
import numpy as np
from basic_image_processor.components.image_converter import GrayScaleConverter
from basic_image_processor.components.morphological_transformers import Erosion
from basic_image_processor.components.threshold import ManualThreshold, AdaptiveGaussThreshold, AutomaticThreshold



class ImageProcessingPipeline:
    def __init__(self):
        self.threshold_type = "adaptive"
        self.manual_value = 127
        self.block_size = 11
        self.C = 2
        self.kernel_size = 3
        self.iterations = 1

    def process(self, frame):
        gray = GrayScaleConverter().apply(frame)

        # Selección dinámica de threshold
        # if self.threshold_type == "manual":
        binary = ManualThreshold().apply(gray, self.manual_value, 255, cv2.THRESH_BINARY)

        # elif self.threshold_type == "adaptive":
        #     binary = AdaptiveGaussThreshold().apply(gray)
        #
        # elif self.threshold_type == "otsu":
        #     binary = AutomaticThreshold().apply(gray)

        # Morph (ajustable)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        binary = Erosion().apply(binary, kernel, self.iterations)

        return binary