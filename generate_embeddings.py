from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pickle
import tqdm
import os

# Define the custom download directory
custom_path = "/globalscratch/ucl/cental/troux/corpus/stsb"
lang = 'fr'
set_type = 'train'
sentence_key = 'sentence2'
saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_{sentence_key}.pkl"
model_name = "Lajavaness/sentence-camembert-base"


concat = True
if concat:
    # concat sentence1 and sentence2 pkl in one pickle
    saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_concat.pkl"
    # load pickles
    key1 = "sentence1"
    key2 = "sentence2"
    saved_embeddings_path1 = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_{key1}.pkl"
    saved_embeddings_path2 = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_{lang}_{set_type}_{key2}.pkl"
    if os.path.exists(saved_embeddings_path):
        print(f"Embeddings already exist at {saved_embeddings_path}. Loading...")
        exit()
    else:
        if os.path.exists(saved_embeddings_path1) and os.path.exists(saved_embeddings_path2):
            print(f"Embeddings already exist at {saved_embeddings_path1} and {saved_embeddings_path2}. Loading...")
            with open(saved_embeddings_path1, "rb") as f:
                id2embedding1 = pickle.load(f)
            with open(saved_embeddings_path2, "rb") as f:
                id2embedding2 = pickle.load(f)
            print(f"Loaded {len(id2embedding1)} and {len(id2embedding2)} embeddings")
            # concat
            id2embedding = {}
            new_id = 0
            for id in id2embedding1.keys():
                text1, embedding1 = id2embedding1[id]
                text2, embedding2 = id2embedding2[id]
                id2embedding[new_id] = text1, embedding1
                new_id += 1
                id2embedding[new_id] = text2, embedding2
                new_id += 1
            # print
            print(f"Concatenated {len(id2embedding1)} and {len(id2embedding2)} embeddings to {len(id2embedding)}")
            # save
            with open(saved_embeddings_path, "wb") as f:
                pickle.dump(id2embedding, f)
            print(f"Embeddings saved to {saved_embeddings_path}")
        else:
            print(f"Embeddings not found at {saved_embeddings_path1} or {saved_embeddings_path2}. Exiting...")
            exit()
        exit()


# Check if the embeddings are already generated
if os.path.exists(saved_embeddings_path):
    print(f"Embeddings already exist at {saved_embeddings_path}. Loading...")
    with open(saved_embeddings_path, "rb") as f:
        id2embedding = pickle.load(f)
    print(f"Loaded {len(id2embedding)} embeddings")
else:
    # Load the French STSB dataset and save it to the specified path
    dataset = load_dataset("stsb_multi_mt", name=lang, cache_dir=custom_path)
    # Load the Sentence Transformer model
    model = SentenceTransformer(model_name)
    id2embedding = {}
    # for id, example in enumerate(dataset["train"]):
    for id, example in tqdm.tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        embedding = model.encode(example[sentence_key]) # sentence1 or sentence2
        id2embedding[id] = (example[sentence_key], embedding)

    # Save the embeddings to a file
    with open(saved_embeddings_path, "wb") as f:
        pickle.dump(id2embedding, f)
    print(f"Embeddings saved to {saved_embeddings_path}")

print("Example embeddings:")
for id, (text, embedding) in id2embedding.items():
    print(f"ID: {id}, Text: {text}, Embedding shape: {embedding.shape}")
    break