import torch
from torch.utils.data import Dataset

class KGNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, config):
        self.config = config
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.enhanced_text  # 使用增强后的文本
        self.targets = dataframe.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.config.MAX_LEN,
            padding='max_length',
            return_token_type_ids=False,  # DistilBERT 不需要 token_type_ids
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
