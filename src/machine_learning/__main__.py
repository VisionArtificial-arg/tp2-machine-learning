# from basic_image_processor.components.image_converter.grayscale_conversion import GrayScaleConverter
# from basic_image_processor.components.threshold import AdaptiveGaussThreshold
# from basic_image_processor.components.morphological_transformers import Erosion
from .components.descriptor_generator.data_set_hu_moments_handler import DatasetHuMomentsHandler

handler = DatasetHuMomentsHandler(
    # labels=["triangle", "rectangle", "5-point-star"],
    shapes_path="./shapes",
    output_path="./generated_files/shapes-hu-moments.csv",
    # grayscale=GrayScaleConverter(),
    # threshold=AdaptiveGaussThreshold(),
    # erosion=Erosion(),
)

handler.generate_hu_moments_file()