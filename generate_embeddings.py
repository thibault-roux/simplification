from datasets import load_dataset

# Define the custom download directory
custom_path = "/globalscratch/ucl/cental/troux/corpus/stsb"

# Load the French STSB dataset and save it to the specified path
dataset = load_dataset("stsb_multi_mt", name="fr", cache_dir=custom_path)

# Print an example
print(dataset["train"][0])
