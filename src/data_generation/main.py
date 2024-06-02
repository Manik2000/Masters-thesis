from src.data_generation.classifier_data import main as main_classifier
from src.data_generation.final_evaluation_data import main as main_final

if __name__ == "__main__":
    window_lengths = [10, 15, 25]
    for window_length in window_lengths:
        print(f"Generating data for window_length: {window_length}")
        main_classifier(window_length)
        print()
    # main_final()
