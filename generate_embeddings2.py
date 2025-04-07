from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pickle
import tqdm
import os
import re

# Define the custom download directory
dataset_name = "asi/wikitext_fr"
if dataset_name == "stsb_multi_mt":
    acronym = "stsb"
elif dataset_name == "asi/wikitext_fr":
    acronym = "wikitext"
custom_path = f"/globalscratch/ucl/cental/troux/corpus/{acronym}"
info_name = 'wikitext-72'
set_type = 'train'
sentence_key = 'paragraph'
split_sentence = True
if split_sentence:
    splitter = "_split"
else:
    splitter = ""
LIMIT_MAX_LENGTH = 999999
if LIMIT_MAX_LENGTH >= 999999:
    uplimit = ""
else:
    uplimit = f"_limit{LIMIT_MAX_LENGTH}"
saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/{acronym}/embeddings_{info_name}_{set_type}_{sentence_key}{splitter}{uplimit}.pkl"
model_name = "Lajavaness/sentence-camembert-base"






concat = False
if concat:
    # concat sentence1 and sentence2 pkl in one pickle
    saved_embeddings_path = f"/globalscratch/ucl/cental/troux/corpus/{acronym}/embeddings_{info_name}_{set_type}_wiki_stsb.pkl"
    # load pickles
    key1 = "concat"
    key2 = "paragraph"
    saved_embeddings_path1 = f"/globalscratch/ucl/cental/troux/corpus/stsb/embeddings_fr_{set_type}_{key1}.pkl"
    saved_embeddings_path2 = f"/globalscratch/ucl/cental/troux/corpus/wikitext/embeddings_{info_name}_{set_type}_{key2}.pkl"
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
    # Load the dataset and save it to the specified path
    dataset = load_dataset(dataset_name, name=info_name, cache_dir=custom_path)
    # Load the Sentence Transformer model
    model = SentenceTransformer(model_name)
    id2embedding = {}
    # for id, example in enumerate(dataset["train"]):
    sent_id = 0
    for id, example in tqdm.tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        # embedding = model.encode(example[sentence_key]) # sentence1 or sentence2 or paragraph
        # id2embedding[id] = (example[sentence_key], embedding)

        # print(example[sentence_key])
        # split by sentence without removing punctuation
        sentences = re.findall(r'[^.]+[.]', example[sentence_key])
        for sentence in sentences:
            if len(sentence) > 3 and len(sentence) < LIMIT_MAX_LENGTH:
                embedding = model.encode(sentence)
                id2embedding[sent_id] = (sentence, embedding)
                sent_id += 1

        

    # Save the embeddings to a file
    with open(saved_embeddings_path, "wb") as f:
        pickle.dump(id2embedding, f)
    print(f"Embeddings saved to {saved_embeddings_path}")

print("Example embeddings:")
for id, (text, embedding) in id2embedding.items():
    print(f"ID: {id}, Text: {text}, Embedding shape: {embedding.shape}")
    break