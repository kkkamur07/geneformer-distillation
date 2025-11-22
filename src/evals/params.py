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