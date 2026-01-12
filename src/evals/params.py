"""
Available Parameters Inspection Script.

A utility script to quickly instantiate the StudentModel architecture and print
the number of trainable parameters. Useful for sanity checking model size configurations.

Output:
    Prints the total count of parameters in the model configuration.

Usage Example:
    ```bash
    python -m src.evals.params
    ```
"""
from src.model.student_model import StudentModel

def main(): 
    
    model = StudentModel(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size = 384,
    )

    model.get_num_parameters()
    print(f"Number of parameters in the model: {model.get_num_parameters()}")
    
if __name__ == "__main__":
    main()