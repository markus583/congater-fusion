import torch
from typing import Dict


from arguments_multi import task_to_keys


class TaskCollator:
    """Implements task-collator to collate the samples in each batch."""

    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # because of padding="longest" this does not work to be done in dataset part.
        return self._encode(batch)

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        tasks = [x["task"] for x in batch]
        # There should be only one task per batch.
        assert len(set(tasks)) == 1

        batched_examples = [x["example"] for x in batch]
        # args = (
        #     (examples[self.sentence1_key],)
        #     if self.sentence2_key is None
        #     else (examples[self.sentence1_key], examples[self.sentence2_key])
        # )
        sentence1_key, sentence2_key = task_to_keys[tasks[0]]
        if tasks[0] == "copa":
            result1 = self.tokenizer(
                [examples["text_a"] for examples in batched_examples],
                [examples["choice1"] for examples in batched_examples],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )
            result2 = self.tokenizer(
                [examples["text_a"] for examples in batched_examples],
                [examples["choice2"] for examples in batched_examples],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )
            result = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
                    result[key] = torch.tensor(result[key])

            result["task"] = tasks[0]
            result["labels"] = torch.tensor([x["label"] for x in batched_examples])
            return result

        batched_args = [
            examples[sentence1_key]
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
            for examples in batched_examples
        ]

        if tasks[0] == "multirc":
            max_length = 348
        else:
            max_length = self.max_seq_length
        if tasks[0] == "record":
            truncation = "only_first"
            max_length = self.max_seq_length * 2
        else:
            truncation = True
        result = self.tokenizer(
            batched_args,
            padding=self.padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
        )

        result["task"] = tasks[0]
        result["labels"] = torch.tensor([x["label"] for x in batched_examples])
        return result.data
