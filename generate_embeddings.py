from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pickle
import tqdm

# Define the custom download directory
custom_path = "/globalscratch/ucl/cental/troux/corpus/stsb"
lang = 'fr'
set_type = 'train'
sentence_key = 'sentence1'
saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_{sentence_key}.pkl"
model_name = "Lajavaness/sentence-camembert-base"

# Load the French STSB dataset and save it to the specified path
dataset = load_dataset("stsb_multi_mt", name=lang, cache_dir=custom_path)

model = SentenceTransformer(model_name)

id2embedding = {}
# for id, example in enumerate(dataset["train"]):
for id, example in tqdm.tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
    embedding = model.encode(example[sentence_key]) # sentence1 or sentence2
    id2embedding[id] = embedding

# Save the embeddings to a file
with open(saved_embeddings_path, "wb") as f:
    pickle.dump(id2embedding, f)
print(f"Embeddings saved to {saved_embeddings_path}")
