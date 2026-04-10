import cv2

from components.image_pipeline import ImageProcessingPipeline
from components.inference.inference_runner import InferenceRunner
from components.inference.stable_predictor import StablePredictor


def main():

    pipeline = ImageProcessingPipeline()

    cv2.namedWindow("Controls")
    # cv2.createTrackbar("Threshold", "Controls", 127, 255, lambda x: None)
    cv2.createTrackbar("Kernel", "Controls", 1, 5, lambda x: None)
    # cv2.createTrackbar("Iterations", "Controls", 1, 10, lambda x: None)

    cap = cv2.VideoCapture(0)
    runner = InferenceRunner(
        model_path="generated_files/model.joblib",
        testing_images_path="./shapes/testing"
    )

    stable = StablePredictor(window=12)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pipeline.manual_value = cv2.getTrackbarPos("Threshold", "Controls")
        pipeline.kernel_size = cv2.getTrackbarPos("Kernel", "Controls") | 1
        pipeline.iterations = cv2.getTrackbarPos("Iterations", "Controls")

        processed = pipeline.process(frame)

        # 🔥 Predicción cruda (puede variar mucho)
        raw_label = runner.predict_from_frame(frame)

        # 🔥 Predicción filtrada (estable)
        label = stable.update(raw_label)

        print("Label:" , label)


        if label is not None:
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Camera", frame)
        cv2.imshow("Processed", processed)

        if cv2.waitKey(1) == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()