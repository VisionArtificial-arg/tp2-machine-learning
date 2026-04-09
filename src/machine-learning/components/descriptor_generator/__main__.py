from data_set_hu_moments_handler import DatasetHuMomentsHandler

def main():
    generator = DatasetHuMomentsHandler(
        shape_path="shapes",
        output_path="generated_files/shapes-hu-moments.csv"
    )
    generator.generate_hu_moments_file()

if __name__ == "__main__":
    main()



