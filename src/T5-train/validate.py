import os
global device
global console
from os.path import dirname
import numpy as np
import pandas as pd
import torch
from rich import console
from torch.cuda import device
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import display_df
from dataset import DataSetClass
from train import train
from torch import cuda
import math
import torch.nn as nn

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console

# define a rich console logger
console = Console(record=True)


def validate(epoch, tokenizer, model, device, loader):
    """
  Function to evaluate model for predictions

  """
    model.eval()
    predictions = []
    actuals = []
    epoch_loss = 0
    validation_loss = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                )
            
            # Compute the loss
            outputs = model(input_ids=ids, attention_mask=mask)
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            
            # Convert generated_ids to float32
            generated_ids = generated_ids.float()
            
            if _ % 10 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
                        
    return predictions, validation_loss