from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from torch import nn 

model_name = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) 

imdb_dataset = load_dataset("imdb")
train_dataset = imdb_dataset["train"]
val_dataset = imdb_dataset["test"] 

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


train_loader = DataLoader(tokenized_train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_val_dataset, batch_size=16)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch: {epoch + 1} train_loss: {train_loss / len(train_loader)}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            predicted = torch.argmax(outputs.logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch: {epoch + 1} val_loss: {val_loss / len(val_loader)}")
    print(f"Epoch: {epoch + 1} Accuracy {correct / total}")


def predict_sentiment(text, model, tokenizer, device):
    encoded_text = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model.eval()
        outputs = model(**encoded_text)
    predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return "positive" if predicted_label == 1 else "negative" 

new_review = "This movie was a complete waste of time. The acting was terrible!"
predicted_sentiment = predict_sentiment(new_review, model, tokenizer, device)
print(f"Predicted Sentiment: {predicted_sentiment}")