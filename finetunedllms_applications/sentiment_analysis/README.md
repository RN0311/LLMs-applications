# IMDB Sentiment Analysis

This use-case fine-tunes DistilBERT on IMDB dataset for sentiment classification.

## Setup
```bash
pip install torch transformers datasets tqdm
```

## Features
- Uses DistilBERT base uncased model
- Trains on IMDB dataset
- Includes validation during training
- Provides sentiment prediction for new reviews


## Model Details
- Architecture: DistilBERT
- Max sequence length: 128
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
