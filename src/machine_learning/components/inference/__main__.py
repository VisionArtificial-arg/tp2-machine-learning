import cv2

from src.machine_learning.components.image_pipeline import ImageProcessingPipeline
from src.machine_learning.components.inference.inference_runner import InferenceRunner
from src.machine_learning.components.inference.stable_predictor import StablePredictor


def main():

    pipeline = ImageProcessingPipeline()

    cv2.namedWindow("Controls")
    cv2.createTrackbar("Threshold", "Controls", 127, 255, lambda x: None)
    cv2.createTrackbar("Kernel", "Controls", 1, 5, lambda x: None)
    # cv2.createTrackbar("Iterations", "Controls", 1, 10, lambda x: None)

    cap = cv2.VideoCapture(0)
    runner = InferenceRunner(
        model_path="generated_files/model.joblib",
        testing_images_path="./shapes/testing",
    )

    stable = StablePredictor(window=12)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # pipeline.manual_value = cv2.getTrackbarPos("Threshold", "Controls")
        pipeline.kernel_size = cv2.getTrackbarPos("Kernel", "Controls") | 1
        # pipeline.iterations = cv2.getTrackbarPos("Iterations", "Controls")

        processed = pipeline.process(frame)

        contours, _ = cv2.findContours(
            processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (
                1000 < area < 100000
            ):  # Ajusta este valor según la resolución de tu cámara
                cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

                label = runner.predict_from_contour(cnt)  # realizar prediccion

                # logica para encontrar en el centro de la silueta la etiqueta que lo representa
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                # dibujar la etiqueta cerca del centro de la figura
                cv2.putText(
                    frame,
                    label,
                    (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Camera", frame)
        cv2.imshow("Processed", processed)

        if cv2.waitKey(1) == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

