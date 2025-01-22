import torch #Provides tensor operations and deep learning functionalities.
from torch.utils.data import Dataset #Base class for custom datasets.
from torch.utils.data import DataLoader #Utility for creating data loader

class GPReviewDataset(Dataset):
    """
    Custom Dataset class for processing reviews and corresponding sentiment targets.

    Attributes:
        reviews (array-like): Collection of review texts.
        targets (array-like): Sentiment labels for the reviews.
        tokenizer (transformers tokenizer): Tokenizer for encoding the review texts.
        max_len (int): Maximum length for tokenized sequences.
    """
    def __init__(self, reviews, targets, tokenizer, max_len):
        """
        Initializes the dataset with reviews, targets, tokenizer, and max sequence length.

        Args:
            reviews (array-like): Collection of review texts.
            targets (array-like): Sentiment labels for the reviews.
            tokenizer (transformers tokenizer): Tokenizer for encoding the review texts.
            max_len (int): Maximum length for tokenized sequences.
        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of reviews in the dataset.

        Returns:
            int: Number of reviews.
        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Retrieves a single data point from the dataset, including tokenized inputs and targets.

        Args:
            item (int): Index of the review to retrieve.

        Returns:
            dict: Contains tokenized input IDs, attention mask, and target label.
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,  # Ensure truncation for long texts
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Creates a DataLoader for batching and shuffling the dataset.

    Args:
        df (DataFrame): DataFrame containing the dataset with 'Text' and 'Sentiment' columns.
        tokenizer (transformers tokenizer): Tokenizer for encoding the review texts.
        max_len (int): Maximum length for tokenized sequences.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = GPReviewDataset(
        reviews=df.Text.to_numpy(),
        targets=df.Sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for debugging and compatibility
        shuffle=True
    )

def __getitem__(self, index):
    """
    Retrieves a single data point from an external dataset and handles errors gracefully.

    Args:
        index (int): Index of the data point to retrieve.

    Returns:
        dict: Contains raw text and the target label.

    Raises:
        Exception: If an error occurs during data retrieval or conversion.
    """
    data = self.data.iloc[index]
    target = data['Sentiment']

    try:
        target = int(target)  # Explicitly cast to int
        return {
            'text': data['Text'],  
            'targets': torch.tensor(target, dtype=torch.long)
        }
    except Exception as e:
        print(f"Error at index {index}: {target}, Error: {e}")
        raise

