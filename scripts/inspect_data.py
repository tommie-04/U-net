"""Load and inspect the IMDb training data."""

from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", split="train")

# Print a summary of the dataset.
print(dataset)

# Print the number of reviews.
print(len(dataset))

# Print the first sample.
print(dataset[0])
