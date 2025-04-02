import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, image_embedding_dim=768, gpt_model="dbddv01/gpt2-french-small"):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.image_projection = nn.Linear(image_embedding_dim, self.gpt2.config.hidden_size)

    def forward(self, image_features, max_length=30):
        projected_features = self.image_projection(image_features).unsqueeze(1)

        # Start with the [BOS] token (or use eos_token for consistency)
        input_ids = torch.full((image_features.shape[0], 1), self.tokenizer.bos_token_id, device=image_features.device, dtype=torch.long)

        # Autoregressive generation
        for _ in range(max_length):
            token_embeddings = self.gpt2.transformer.wte(input_ids)
            inputs_embeds = torch.cat([projected_features, token_embeddings], dim=1)
            outputs = self.gpt2(inputs_embeds=inputs_embeds)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all generated tokens are EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return input_ids





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
            generated_ids = model(image_features, max_length=input_ids.size(1))  # Generate caption

            loss = criterion(generated_ids[:, 1:].reshape(-1), input_ids[:, 1:].reshape(-1))  # Shift target for prediction
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_data = get_training_data(lang='fr', set_type='train')
    tokenizer = GPT2Tokenizer.from_pretrained("dbddv01/gpt2-french-small")
    tokenizer.pad_token = tokenizer.eos_token

    # Create Dataset & DataLoader
    dataset = CaptionDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize Model, Optimizer & Loss
    model = ImageCaptioningModel(image_embedding_dim=768).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train the Model
    train(model, dataloader, optimizer, criterion, device, epochs=200)

    # Save the trained model
    torch.save(model.state_dict(), "image_captioning_model.pth")
    print("Model saved!")
