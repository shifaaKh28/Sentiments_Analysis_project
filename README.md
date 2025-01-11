DL_project/img1.png
# Sentiment Analysis Project

This project implements a sentiment analysis pipeline using PyTorch and transformers. The model is designed to classify textual data into sentiments such as Positive, Neutral, and Negative. This README provides an overview of the project structure, functionality, and usage.

## Project Overview

### Key Features:
- **Data Preprocessing**: Standardizes sentiment labels and tokenizes text using a pre-trained DistilBERT tokenizer.
- **Model Training**: Implements both advanced transformer-based models and simple neural networks for sentiment classification.
- **Evaluation**: Includes metrics such as accuracy, precision, recall, and F1-score. Provides visualization tools like confusion matrices and training curves.
- **Prediction**: Allows for sentiment prediction on new textual inputs with class probabilities.

## File Structure

### Main Files:

1. **main.py**: The primary script to run the end-to-end pipeline (data preprocessing, training, evaluation, and prediction).
2. **data_loader.py**: Contains utilities to create PyTorch DataLoader for batching and shuffling data.
3. **preprocessing.py**: Handles dataset loading, cleaning, and preprocessing (e.g., sentiment mapping, tokenization).
4. **models.py**: Defines models including a transformer-based classifier and a simple neural network.
5. **training.py**: Implements the training loop for the models.
6. **evaluation.py**: Provides functions for model evaluation and metrics calculation.
7. **visualization.py**: Includes functions for visualizing sentiment distributions and training metrics.

## Installation

### Prerequisites:
- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- pandas
- seaborn
- matplotlib
- tqdm

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model:
1. Place your dataset in the root directory as `data.csv`.
2. Run the `main.py` script:
   ```bash
   python main.py
   ```

### Predictions:
Use the `main.py` script to generate predictions for sample texts. Modify the `sample_texts` list in the script to include your input data.

### Expected Output:
- Training and validation metrics logged for each epoch.
- Confusion matrix and evaluation metrics displayed for the test dataset.
- Predictions with probabilities for sample texts printed in the console.

## Model Details

### Transformer-Based Model:
- **Architecture**: Uses a pre-trained DistilBERT model with a fully connected classification head.
- **Hyperparameters**:
  - Learning Rate: 1e-5
  - Epochs: 20
  - Batch Size: 16
  - Max Sequence Length: 50

### Metrics:
- **Accuracy**: Percentage of correctly classified samples.
- **Precision**: Fraction of true positive predictions among all positive predictions.
- **Recall**: Fraction of true positive predictions among all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

## Visualizations
- **Training Metrics**: Plots for training/validation loss and accuracy over epochs.
- **Confusion Matrix**: Heatmap for evaluating model predictions on the test dataset.
- **Sentiment Distribution**: Bar chart of sentiment classes in the dataset.

