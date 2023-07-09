from datasets import load_dataset
from transformers import AutoTokenizer


def get_average_lengths(dataset, tokenizer):
    lengths = []
    for example in dataset:
        # example["question_answer"] = f'{example["question"]} {example["answer"]}'
        inputs = tokenizer(
            example["passage"],
            example["query"],
            truncation=True,
            padding="max_length",
        )
        lengths.append(sum(inputs["attention_mask"]))
    average_length = sum(lengths) / len(lengths)
    return average_length


# Load the MultIRC dataset from SuperGLUE
dataset = load_dataset("super_glue", "record")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Get the average lengths of the samples for each split
train_avg_length = get_average_lengths(dataset["train"], tokenizer)
dev_avg_length = get_average_lengths(dataset["validation"], tokenizer)
test_avg_length = get_average_lengths(dataset["test"], tokenizer)

# Print the average lengths for each split
print("Average Lengths:")
print("Train Split:", train_avg_length)
print("Dev Split:", dev_avg_length)
print("Test Split:", test_avg_length)
