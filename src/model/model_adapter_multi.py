# from .utils import get_model
from operator import attrgetter

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, RobertaPreTrainedModel
from transformers.adapters import AutoAdapterModel
from transformers.adapters.training import setup_adapter_training
import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class MultitaskAdapterModel(RobertaPreTrainedModel):
    def __init__(self, encoder, shared_params, taskmodels_dict):
        super().__init__(PretrainedConfig())

        self.encoder = encoder
        self.shared_params = shared_params
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(
        cls,
        model_name,
        model_type_dict,
        model_config_dict,
        dataset_cls,
        own_params=None,
        freeze_base_model: bool = True,
    ):
        shared_encoder = None
        taskmodels_dict = {}

        for (task_name, model_type), dataset in zip(
            model_type_dict.items(), dataset_cls
        ):
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            model.add_adapter(adapter_name=task_name, config="pfeiffer")

            if shared_encoder is None:
                shared_encoder = model.base_model
                if own_params is not None:
                    shared_params = set(model.base_model.state_dict().keys()) - set(
                        own_params
                    )                    
                else:
                    shared_params = set(model.base_model.state_dict().keys())
                print(len(shared_params))
            else:
                for param_name, param in model.base_model.named_parameters():
                    if param_name in shared_params and "adapter" not in param_name:
                        # print(param_name)
                        weights = rgetattr(shared_encoder, param_name)
                        # set the shared param to the new model's param
                        rsetattr(model.base_model, param_name, weights)

            if dataset.multiple_choice:
                model.add_multiple_choice_head(
                    task_name, num_choices=2, layers=1, overwrite_ok=True
                )
            else:
                model.add_classification_head(
                    task_name,
                    num_labels=dataset.num_labels,
                    overwrite_ok=True,
                    id2label={i: v for i, v in enumerate(dataset.label_list)}
                    if not dataset.is_regression
                    else None,
                )
            if freeze_base_model:
                model.train_adapter([task_name])
            model.set_active_adapters(task_name)
            taskmodels_dict[task_name] = model

        return cls(
            encoder=shared_encoder,
            shared_params=shared_params,
            taskmodels_dict=taskmodels_dict,
        )

    def forward(self, **kwargs):
        task = kwargs.pop("task")
        return self.taskmodels_dict[task](**kwargs)


    def print_parameter_info(self):
        print("Shared Parameters:")
        print("==================")
        shared_param_count = 0
        shared_trainable_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name in self.shared_params and "adapters" not in param_name:
                shared_param_count += param.numel()
                trainable = param.requires_grad
                shared_trainable_count += param.numel() if trainable else 0
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nNon-Shared Parameters:")
        print("======================")
        non_shared_param_count = 0
        non_shared_trainable_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name not in self.shared_params and "adapters" not in param_name:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                non_shared_trainable_count += param.numel() if trainable else 0
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )
        print("\nNon-shared Adapter Parameters:")
        print("======================")
        for task in self.taskmodels_dict:
            for param_name, param in self.taskmodels_dict[task].named_parameters():     
                if "adapters" in param_name:
                    non_shared_param_count += param.numel()
                    trainable = param.requires_grad
                    non_shared_trainable_count += param.numel() if trainable else 0
                    print(
                        f"Parameter: {param_name}, params: {param.numel()}, Trainable: {param.requires_grad}"
                    )
            
        print("\nNon-shared Head Parameters:")
        print("======================")
        for param_name, param in self.named_parameters():
            if "heads" in param_name:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                non_shared_trainable_count += param.numel() if trainable else 0
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nParameter Counts:")
        print("=================")
        print("\nShared Parameter Count:")
        print("Total Parameters:", shared_param_count)
        print("Trainable Parameters:", shared_trainable_count)
        print("Non-Trainable Parameters:", shared_param_count - shared_trainable_count)
        print("Trainable Proportion:", shared_trainable_count / shared_param_count)
        print(
            "Non-Trainable Proportion:",
            (shared_param_count - shared_trainable_count) / shared_param_count,
        )

        print("=================")
        print("\nNon-Shared Parameter Count:")
        print("Total Parameters:", non_shared_param_count)
        print("Trainable Parameters:", non_shared_trainable_count)
        print(
            "Non-Trainable Parameters:", non_shared_param_count - non_shared_trainable_count
        )
        print("Trainable Proportion:", non_shared_trainable_count / non_shared_param_count)
        print(
            "Non-Trainable Proportion:",
            (non_shared_param_count - non_shared_trainable_count) / non_shared_param_count,
        )

        print("\nTotal Parameter Count:")
        print("=================")
        print("Total Parameters:", shared_param_count + non_shared_param_count)
        print("Trainable Parameters:", shared_trainable_count + non_shared_trainable_count)
        print(
            "Non-Trainable Parameters:",
            shared_param_count
            + non_shared_param_count
            - shared_trainable_count
            - non_shared_trainable_count,
        )
        print(
            "Trainable Proportion:",
            (shared_trainable_count + non_shared_trainable_count)
            / (shared_param_count + non_shared_param_count),
        )
        print(
            "Non-Trainable Proportion:",
            (
                shared_param_count
                + non_shared_param_count
                - shared_trainable_count
                - non_shared_trainable_count
            )
            / (shared_param_count + non_shared_param_count),
        )


if __name__ == "__main__":
    from transformers import AutoConfig

    model_name = "roberta-base"
    model_config_dict = {
        "cola": AutoConfig.from_pretrained(model_name, num_labels=2),
        "sst2": AutoConfig.from_pretrained(model_name, num_labels=2),
        "mrpc": AutoConfig.from_pretrained(model_name, num_labels=2),
        # "sts-b": AutoConfig.from_pretrained(model_name, num_labels=1),
        # "qqp": AutoConfig.from_pretrained(model_name, num_labels=2),
        # "mnli": AutoConfig.from_pretrained(model_name, num_labels=3),
        # "qnli": AutoConfig.from_pretrained(model_name, num_labels=2),
        # "rte": AutoConfig.from_pretrained(model_name, num_labels=2),
        # "wnli": AutoConfig.from_pretrained(model_name, num_labels=2),
    }
    model_type_dict = {
        task_name: AutoAdapterModel for task_name in model_config_dict.keys()
    }
    # OWN_PARAMS = set({"encoder.layer.11.output.LayerNorm.bias"})

    model = MultitaskAdapterModel.create(
        model_name=model_name,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict,
        # own_params=OWN_PARAMS,
    )

    model.print_parameter_info()
