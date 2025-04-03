import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer



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
def get_training_data(lang='fr', set_type='train', sentence_key='sentence1'):
    '''
    Load the training data from the saved embeddings
    Args:
        lang: str, language of the dataset
        set_type: str, type of the dataset (train, dev, test)
        sentence_key: str, key of the sentence in the dataset
    Returns:
        id2embedding: dict, mapping of sentence ID to a tuple (text, embeddings)
    '''
    saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_{sentence_key}.pkl"
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
        self.data = list(data.items())  # Convert dict to list of tuples (id, (text, embedding))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data) - 1 # DON'T FORGET TO REMOVE THE -1 AFTER EXPERIMENTING -------------------------------------------- 

    def __getitem__(self, idx):
        text, embedding = self.data[idx][1]  # Get caption & embedding
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        # Tokenize caption
        tokenized_caption = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenized_caption["input_ids"].squeeze(0)  # Remove batch dim
        attention_mask = tokenized_caption["attention_mask"].squeeze(0)

        return embedding_tensor, input_ids, attention_mask

# Training Function
def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0

        for image_features, input_ids, _ in dataloader:  # Ignore attention_mask
            image_features, input_ids = image_features.to(device), input_ids.to(device)

            optimizer.zero_grad()

            logits = model(image_features, input_ids[:, :-1])  # Get logits, not tokens

            # Shift labels left (ignore BOS, predict next tokens)
            seq_len = input_ids.size(1) - 1  # Expected target sequence length
            loss = criterion(logits[:, :seq_len, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))


            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")



        # Infer on the last text of the dataset
        text, embedding = train_data[len(train_data) - 1]
        tokenized_text = tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        output = model(torch.tensor(embedding).unsqueeze(0).to(device), input_ids)
        predicted_text = tokenizer.decode(output.argmax(-1).squeeze().tolist())
        print(f"Predicted: {predicted_text}")
        print(f"Actual: {text}")
        print("--------------------")



# Main Script
if __name__ == "__main__":
    print("Launching...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_data = get_training_data(lang='fr', set_type='train', sentence_key='concat')
    tokenizer = GPT2Tokenizer.from_pretrained("asi/gpt-fr-cased-base")
    tokenizer.pad_token = tokenizer.eos_token

    # Create Dataset & DataLoader
    dataset = CaptionDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Model, Optimizer & Loss
    model = ImageCaptioningModel(image_embedding_dim=768).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train the Model
    train(model, dataloader, optimizer, criterion, device, epochs=200)

    # Save the trained model
    torch.save(model.state_dict(), "results/image_captioning_model.pth")
    print("Model saved!")
