from data_loader import create_data_loader
from preprocessing import load_dataset, preprocess_dataset, tokenizer

df = load_dataset("data.csv")
df, df_filtered = preprocess_dataset(df)

MAX_LEN = 50
BATCH_SIZE = 8

train_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

for batch in train_data_loader:
    print(batch.keys())
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['targets'].shape)
    break
