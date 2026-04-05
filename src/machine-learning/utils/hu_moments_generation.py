import math
import cv2

from basic_image_processor.components.image_converter.grayscale_conversion import GrayScaleConverter
from basic_image_processor.components.morphological_transformers import Erosion
from basic_image_processor.components.threshold import AdaptiveGaussThreshold


def hu_moments_of_file(filename: str):
    image = cv2.imread(filename) # read image

    gray = GrayScaleConverter().apply(image) # applies grayscale

    binary = AdaptiveGaussThreshold().apply(gray) # apply adaptive threshold

    binary = 255 - binary # invert values

    binary = Erosion().apply(binary) # applies Erosion

    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv2.contourArea)

    # get moments
    moments = cv2.moments(shape_contour)
    hu = cv2.HuMoments(moments)

    for i in range(7):
        hu[i] = -1 * math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))

    return hu