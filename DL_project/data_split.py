import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df):
    # Features and target
    X = df['Clean_Text']
    y = df['Simplified_Sentiment']

    # Map target labels to numbers
    label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    y = y.map(label_mapping)

    # Perform the split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    # Combine features and labels for saving
    train = pd.DataFrame({'Clean_Text': X_train, 'Simplified_Sentiment': y_train})
    val = pd.DataFrame({'Clean_Text': X_val, 'Simplified_Sentiment': y_val})
    test = pd.DataFrame({'Clean_Text': X_test, 'Simplified_Sentiment': y_test})

    # Save each split to a CSV file
    train.to_csv('train.csv', index=False)
    val.to_csv('val.csv', index=False)
    test.to_csv('test.csv', index=False)

    print("\nSplits saved as 'train.csv', 'val.csv', and 'test.csv'.")
