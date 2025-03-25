import os
from os.path import dirname
import numpy as np
import pandas as pd
import torch
# from rich import console
global device
from torch.cuda import device
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import display_df
from dataset import DataSetClass
from torch import cuda
import math
import torch.nn as nn
from validate import validate

global console
from rich.console import Console
from tqdm import tqdm


# define a rich console logger
console = Console(record=True)


# Define the T5Trainer
def T5Trainer(test_dataframe, source_text, target_text, model_params, dataset_name ,output_dir="./"):
    """
    T5 trainer

    """
    output_dir= "./outputs/"+str(dataset_name)+"/"
    path = "./model_files_19_java/"
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(path)

    #loading the model

    
    model = T5ForConditionalGeneration.from_pretrained(path)
    model = model.to(device)

    # Creation of Dataset and Dataloader
    val_dataset = test_dataframe.reset_index(drop=True)
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    val_set = DataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        )
    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    val_loader = DataLoader(val_set, **val_params)
        
    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals, val_loss = validate(epoch, tokenizer, model, device, val_loader) 
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions_eclipse_tfidf"+".csv"))
    print("predictions_complete")
 
    
# Define the common model parameters
common_model_params = {
    "MODEL": "t5-base",  
    "TRAIN_BATCH_SIZE": 8,  
    "VALID_BATCH_SIZE": 8,  
    "LEARNING_RATE": 1e-4,  
    "MAX_SOURCE_TEXT_LENGTH": 512,  
    "MAX_TARGET_TEXT_LENGTH": 128, 
    "SEED": 42,
}

# Define a list of epochs to run
global epochs_to_run
epochs_to_run = [1]

if __name__ == "__main__":
    datasets = ['java']
    
    for name in datasets:
        test_df = pd.read_csv(f"")
                
        print(f"For dataset ---- {name}")
        device = 'cuda' if cuda.is_available() else 'cpu'
        
        for epoch in epochs_to_run:
            model_params = {**common_model_params, "TRAIN_EPOCHS": epoch, "VAL_EPOCHS": epoch}
            
            T5Trainer(
                test_dataframe=test_df,
                source_text="Keyword",
                model_params=model_params,
                output_dir="outputs",  # You might want to change the output directory
                dataset_name=name,
            )
