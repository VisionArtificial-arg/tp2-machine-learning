import math
import cv2
from cv2 import Mat

from basic_image_processor.components.image_converter.grayscale_conversion import GrayScaleConverter
from basic_image_processor.components.morphological_transformers import Erosion
from basic_image_processor.components.threshold import AdaptiveGaussThreshold

def compute_hu_moments(image):
    gray = GrayScaleConverter().apply(image)
    binary = AdaptiveGaussThreshold().apply(gray)
    binary = 255 - binary

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = Erosion().apply(binary, kernel=kernel, iterations=1)

    # contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)

    # momentos
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m)

    # evitar log(0)
    eps = 1e-12
    for i in range(7):
        v = hu[i][0]
        if abs(v) < eps:
            hu[i][0] = 0.0
        else:
            hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

    return hu

def hu_moments_of_file(filename):
    return compute_hu_moments(cv2.imread(filename))

def hu_moments_of_frame(frame):
    return compute_hu_moments(frame)