import torch
from torch.utils.data import Dataset


class DataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for fine-tuning the model

    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_len)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        # t5 input and output
        source_text = str(self.source_text[index])

        # cleaning data to ensure data is in string type
        source_text = " ".join(source_text.split())


        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()


        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
        }

