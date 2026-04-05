from components.training.trainer import DecisionTreeTrainer

def main():
    trainer = DecisionTreeTrainer(
        dataset_path="generated_files/shapes-hu-moments.csv",
        model_output_path="generated_files/model.joblib"
    )

    model = trainer.train()
    trainer.save(model)

if __name__ == "__main__":
    main()