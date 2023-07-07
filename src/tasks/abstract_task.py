"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools

import torch
import evaluate
from typing import Callable, Dict, Mapping, List
import logging
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from transformers import EvalPrediction
import numpy as np


logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer"),
}


import abc
from typing import Dict, Callable, Mapping


class TaskDataset:
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name
        if self.name.lower() in GLUE_DATASETS:
            self.dataset_name = "glue"
        elif self.name.lower() in SUPERGLUE_DATASETS:
            self.dataset_name = "super_glue"
        self.metrics = evaluate.load(self.dataset_name, self.name)
        self.multiple_choice = self.name in ["copa"]
        self.is_regression = self.name == "stsb"
        self.split_to_data_split = {
            "train": "train",
            "validation": "validation",
            "test": "train",
        }
        self.sentence1_key = None
        self.sentence2_key = None

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.dataset_name, self.name, split=split)

    def get_dataset(self, split, n_obs=None):
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(split=mapped_split)
        if n_obs != -1:
            dataset = dataset.select(range(n_obs))
        if split in ["train", "test"]:
            if self.name in ["stsb", "record", "cb"]:
                dataset_dict = dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=42
                )
            else:
                dataset_dict = dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=42, stratify_by_column="label"
                )
            if split == "train":
                dataset = dataset_dict["train"]
            elif split == "test":
                dataset = dataset_dict["test"]
        else:
            if n_obs != -1:
                dataset = dataset.select(range(n_obs))

        if self.name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice:
            self.label_list = dataset.features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        self.sentence1_key, self.sentence2_key = task_to_keys[self.name]

        if not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        return dataset.map(
            functools.partial(self.preprocess_function),
            remove_columns=dataset.column_names,
            load_from_cache_file=True,
        )
        
    def preprocess_function(self, examples):
        # WSC
        if self.data_args.task_name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(
                examples["text"], examples["span2_index"], examples["span2_text"]
            ):
                if self.data_args.template_id == 0:
                    examples["span2_word_text"].append(span2_word + ": " + text)
                elif self.data_args.template_id == 1:
                    words_a = text.split()
                    words_a[span2_index] = "*" + words_a[span2_index] + "*"
                    examples["span2_word_text"].append(" ".join(words_a))

        # WiC
        if self.data_args.task_name == "wic":
            examples["processed_sentence1"] = []
            if self.data_args.template_id == 1:
                self.sentence2_key = "processed_sentence2"
                examples["processed_sentence2"] = []
            for sentence1, sentence2, word, start1, end1, start2, end2 in zip(
                examples["sentence1"],
                examples["sentence2"],
                examples["word"],
                examples["start1"],
                examples["end1"],
                examples["start2"],
                examples["end2"],
            ):
                if self.data_args.template_id == 0:  # ROBERTA
                    examples["processed_sentence1"].append(
                        f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?"
                    )
                elif self.data_args.template_id == 1:  # BERT
                    examples["processed_sentence1"].append(word + ": " + sentence1)
                    examples["processed_sentence2"].append(word + ": " + sentence2)

        # MultiRC
        if self.data_args.task_name == "multirc":
            examples["question_answer"] = []
            for question, answer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {answer}")

        # COPA
        if self.data_args.task_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(
                examples["text_a"],
                examples["choice1"],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )
            result2 = self.tokenizer(
                examples["text_a"],
                examples["choice2"],
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
            return result

        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key is None
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        if self.data_args.task_name == "multirc":
            max_length = self.max_seq_length * 2
        else:
            max_length = self.max_seq_length
        print("max_length", max_length)
        result = self.tokenizer(
            *args, padding=self.padding, max_length=max_length, truncation=True
        )

        return result

    # def preprocessor(self, examples):
    #     if self.name == "wsc":
    #         examples["span2_word_text"] = []
    #         for text, span2_index, span2_word in zip(
    #             examples["text"], examples["span2_index"], examples["span2_text"]
    #         ):
    #             words_a = text.split()
    #             words_a[span2_index] = "*" + words_a[span2_index] + "*"
    #             examples["span2_word_text"].append(" ".join(words_a))
    #     # WSC
    #     if self.name == "wsc":
    #         examples["span2_word_text"] = []
    #         for text, span2_index, span2_word in zip(
    #             examples["text"], examples["span2_index"], examples["span2_text"]
    #         ):
    #             words_a = text.split()
    #             words_a[span2_index] = "*" + words_a[span2_index] + "*"
    #             examples["span2_word_text"].append(" ".join(words_a))

    #     # WiC
    #     if self.name == "wic":
    #         examples["processed_sentence1"] = []
    #         self.sentence2_key = "processed_sentence2"
    #         examples["processed_sentence2"] = []
    #         for sentence1, sentence2, word, start1, end1, start2, end2 in zip(
    #             examples["sentence1"],
    #             examples["sentence2"],
    #             examples["word"],
    #             examples["start1"],
    #             examples["end1"],
    #             examples["start2"],
    #             examples["end2"],
    #         ):
    #             examples["processed_sentence1"].append(word + ": " + sentence1)
    #             examples["processed_sentence2"].append(word + ": " + sentence2)

    #     # MultiRC
    #     if self.name == "multirc":
    #         examples["question_answer"] = []
    #         for question, answer in zip(examples["question"], examples["answer"]):
    #             examples["question_answer"].append(f"{question} {answer}")
        
    #     if self.name == "copa":
    #         examples["text_a"] = []
    #         for premise, question in zip(examples["premise"], examples["question"]):
    #             joiner = "because" if question == "cause" else "so"
    #             text_a = f"{premise} {joiner}"
    #             examples["text_a"].append(text_a)


    #     # args tuple to dict

    #     examples["task"] = self.name
    #     return examples


def build_compute_metrics_fn(
    task_names: List[str],
) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def compute_metrics(self, p: EvalPrediction):
        print("compute metrics now")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        print(preds.sum())

        # if self.data_args.task_name == "record":
        #     return self.record_compute_metrics(p)

        if self.data_args.task_name == "multirc":
            from sklearn.metrics import f1_score

            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.task_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def tasks_metrics(task) -> Dict:
        return functools.partial(
            compute_metrics,
            metrics=TaskDataset(task).metrics,
        )

    return {task: tasks_metrics(task) for task in task_names}
