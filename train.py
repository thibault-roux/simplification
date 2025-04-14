import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import jiwer


# Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, image_embedding_dim=768, gpt_model="asi/gpt-fr-cased-base"):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.image_projection1 = nn.Linear(image_embedding_dim, image_embedding_dim)
        self.image_projection2 = nn.Linear(image_embedding_dim, image_embedding_dim)
        self.image_projection3 = nn.Linear(image_embedding_dim, self.gpt2.config.hidden_size)

    def forward(self, image_features, input_ids=None, max_length=30):
        projected_features = self.image_projection1(image_features).unsqueeze(1)
        projected_features = self.image_projection2(projected_features)
        projected_features = self.image_projection3(projected_features)

        if input_ids is None:  # If no input tokens, generate from scratch
            input_ids = torch.full(
                (image_features.shape[0], 1), self.tokenizer.bos_token_id,
                device=image_features.device, dtype=torch.long
            )

        token_embeddings = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([projected_features, token_embeddings], dim=1)
        outputs = self.gpt2(inputs_embeds=inputs_embeds)  # Now returns logits

        return outputs.logits  # Return logits instead of argmax tokens





# Function to Load Data
def get_training_data(acronym='stsb', info_name='fr', set_type='train', sentence_key='sentence1', adder='', fullconcat=False):
    '''
    Load the training data from the saved embeddings
    Args:
        info_name: str, Language or wikitext type of the dataset
        set_type: str, type of the dataset (train, dev, test)
        sentence_key: str, key of the sentence in the dataset (could be sentence1, sentence2, paragraph)
    Returns:
        id2embedding: dict, mapping of sentence ID to a tuple (text, embeddings)
    '''
    if fullconcat:
        saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/{acronym}/embeddings_{info_name}_{set_type}_wiki_stsb{adder}.pkl"
    else:
        saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/{acronym}/embeddings_{info_name}_{set_type}_{sentence_key}{adder}.pkl"
    if os.path.exists(saved_embeddings_path):
        print(f"Embeddings already exist at {saved_embeddings_path}. Loading...")
        with open(saved_embeddings_path, "rb") as f:
            id2embedding = pickle.load(f)
        print(f"Loaded {len(id2embedding)} embeddings")
        return id2embedding
    else:
        raise FileNotFoundError(f"Embeddings not found at {saved_embeddings_path}. Run generate_embeddings.py first.")


# Custom Dataset Class
class CaptionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=30):
        # self.data = list(data.items())  # Convert dict to list of tuples (id, (text, embedding)) # deprecated due to change in data format
        # data # list of tuples (text, embedding))
        self.data = [(idx, item) for idx, item in enumerate(data)]  # Add index to each item, transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, embedding = self.data[idx][1]  # Get caption & embedding
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        # Tokenize caption
        tokenized_caption = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenized_caption["input_ids"].squeeze(0)  # Remove batch dim
        attention_mask = tokenized_caption["attention_mask"].squeeze(0)

        return embedding_tensor, input_ids, attention_mask

# Training Function
def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    model.to(device)

    # if results/best.txt exists, load it
    if os.path.exists("results/best.txt"):
        with open("results/best.txt", "r") as f:
            lines = f.readlines()
            best_eval_loss = float(lines[-1].split(",")[1].split(":")[1])
            # Load the model state
            model.load_state_dict(torch.load("results/models/emb_captioning_best_model.pth"))
            # Load the optimizer state
            optimizer.load_state_dict(torch.load("results/emb_captioning_best_optimizer.pth"))
            print(f"Loaded best model with loss {best_eval_loss:.4f}")
    else:
        best_eval_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0

        idx_batch = 0
        # for image_features, input_ids, _ in dataloader:  # Ignore attention_mask
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            idx_batch += 1
            image_features, input_ids, _ = batch
            image_features, input_ids = image_features.to(device), input_ids.to(device)

            optimizer.zero_grad()

            logits = model(image_features, input_ids[:, :-1])  # Get logits, not tokens

            # Shift labels left (ignore BOS, predict next tokens)
            seq_len = input_ids.size(1) - 1  # Expected target sequence length
            loss = criterion(logits[:, :seq_len, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), f"results/models/emb_captioning_epoch_{epoch + 1}.pth")
        print(f"Model saved at epoch {epoch + 1}")

        # Evaluate the model
        eval_loss = evaluate(model, valid_dataloader, device)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), "results/models/emb_captioning_best_model.pth")
            torch.save(optimizer.state_dict(), "results/models/emb_captioning_best_optimizer.pth")
            print(f"Best model saved with loss {best_eval_loss:.4f}")
            # Save the best loss
            with open("results/best.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}, Loss: {best_eval_loss:.4f}\n")
        else:
            print(f"Model not improved, current best loss: {best_eval_loss:.4f}")


        model.train()


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            image_features, input_ids, _ = batch
            image_features, input_ids = image_features.to(device), input_ids.to(device)

            logits = model(image_features, input_ids[:, :-1])
            seq_len = input_ids.size(1) - 1
            loss = criterion(logits[:, :seq_len, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_best_model(model, test_data, tokenizer, device):
    # Load the best model
    model.load_state_dict(torch.load("results/models/emb_captioning_best_model.pth"))
    model.to(device)

    # scores
    wer_scores = []
    cer_scores = []
    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model by computing word error rate (WER) between predictions and ground truth
    for batch in tqdm(test_data, desc="Evaluating Best Model", unit="batch"):
        image_features, input_ids, _ = batch
        image_features, input_ids = image_features.to(device), input_ids.to(device)

        # Generate predictions
        with torch.no_grad():
            logits = model(image_features, input_ids[:, :-1])
            predicted_ids = torch.argmax(logits[:, -1, :], dim=-1)

        # Decode the predicted and ground truth captions
        predicted_captions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        ground_truth_captions = tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)

        # Compute WER or any other evaluation metric here
        for pred, gt in zip(predicted_captions, ground_truth_captions):
            wer = jiwer.wer(gt, pred)
            cer = jiwer.cer(gt, pred)
            wer_scores.append(wer)
            cer_scores.append(cer)

    # Compute average WER and CER
    wer_avg = sum(wer_scores) / len(wer_scores)
    cer_avg = sum(cer_scores) / len(cer_scores)

    # Print the results
    print(f"Average WER: {wer_avg:.4f}")
    print(f"Average CER: {cer_avg:.4f}")
    # Save the results
    with open("results/evaluation_results.txt", "w") as f:
        f.write(f"Average WER: {wer_avg:.4f}\n")
        f.write(f"Average CER: {cer_avg:.4f}\n")
    # Save the predictions
    with open("results/predictions.txt", "w") as f:
        for pred, gt in zip(predicted_captions, ground_truth_captions):
            f.write(f"Predicted: {pred}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write("\n")        




# Main Script
if __name__ == "__main__":
    print("Launching...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_data = get_training_data(acronym='wikitext', info_name='wikitext-72', set_type='train', sentence_key='paragraph', adder='_split', fullconcat=False)
    print("before:", type(train_data)) # dict
    # print first element of the dict
    print("first element:", list(train_data.items())[0])
    # split in train, valid and test data with train_test_split
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print("after:", type(train_data)) # list
    # print first element of the list
    print("first element:", train_data[0])

    valid_data, test_data = train_test_split(valid_data, test_size=0.1, random_state=42)
    
    tokenizer = GPT2Tokenizer.from_pretrained("asi/gpt-fr-cased-base")
    tokenizer.pad_token = tokenizer.eos_token

    # Create Dataset & DataLoader
    train_data = CaptionDataset(train_data, tokenizer)
    valid_data = CaptionDataset(valid_data, tokenizer)
    test_data = CaptionDataset(test_data, tokenizer)
    train_data = DataLoader(train_data, batch_size=100, shuffle=True)
    valid_data = DataLoader(valid_data, batch_size=100, shuffle=False)
    test_data = DataLoader(test_data, batch_size=100, shuffle=False)

    # Initialize Model, Optimizer & Loss
    model = ImageCaptioningModel(image_embedding_dim=768).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train the Model
    train(model, train_data, valid_data, optimizer, criterion, device, epochs=200)
