from components.inference.inference_runner import InferenceRunner


def main():
    runner = InferenceRunner(
        model_path="generated_files/model.joblib",
        testing_images_path="./shapes/testing"
    )

    runner.run()


if __name__ == "__main__":
    main()