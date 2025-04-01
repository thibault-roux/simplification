import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ImageCaptioningModel(nn.Module):
    def __init__(self, image_embedding_dim=512, gpt_model="gpt2"):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model)  
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)

        # Fully connected layer to project image feature into GPT hidden space
        self.image_projection = nn.Linear(image_embedding_dim, self.gpt2.config.hidden_size)  

    def forward(self, image_features, input_ids):
        """
        image_features: Tensor of shape (batch_size, image_embedding_dim)
        input_ids: Tokenized input text (batch_size, sequence_length)
        """
        # Project image feature to GPT hidden dimension
        projected_features = self.image_projection(image_features).unsqueeze(1)  # (batch, 1, hidden_dim)

        # Get GPT token embeddings
        token_embeddings = self.gpt2.transformer.wte(input_ids)  # (batch, seq_len, hidden_dim)

        # Concatenate image embedding at the beginning of the sequence
        inputs_embeds = torch.cat([projected_features, token_embeddings], dim=1)  

        # Generate output logits
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        return outputs.logits  # (batch, seq_len + 1, vocab_size)

    def generate_caption(self, image_features, max_length=20):
        """
        Generate a caption given an image feature vector.
        """
        image_features = torch.tensor(image_features).unsqueeze(0)  # Convert to tensor and add batch dim
        projected_features = self.image_projection(image_features).unsqueeze(1)  # (1, 1, hidden_dim)

        # Start with BOS token (GPT-2 uses "<|endoftext|>" as BOS)
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])

        # Autoregressive generation
        for _ in range(max_length):
            token_embeddings = self.gpt2.transformer.wte(input_ids)  # Get token embeddings
            inputs_embeds = torch.cat([projected_features, token_embeddings], dim=1)

            outputs = self.gpt2(inputs_embeds=inputs_embeds)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

            # Stop if EOS token is generated
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Decode generated token sequence
        caption = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return caption


# Example Usage
image_embedding = torch.randn(1, 512)  # Simulated 512-d vector from a CNN/ViT
print(image_embedding)
model = ImageCaptioningModel(image_embedding_dim=512)

# Generate a caption
caption = model.generate_caption(image_embedding.squeeze(0).tolist())
print("Generated Caption:", caption)
