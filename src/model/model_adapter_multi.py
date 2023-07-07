# from .utils import get_model
from operator import attrgetter

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, RobertaPreTrainedModel
from transformers.adapters import AutoAdapterModel


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

            if shared_encoder is None:
                shared_encoder = model.base_model
                if own_params is not None:
                    shared_params = (
                        set(model.base_model.state_dict().keys()) - own_params
                    )
                else:
                    shared_params = set(model.base_model.state_dict().keys())
                print(len(shared_params))
            else:
                for param_name, param in model.base_model.named_parameters():
                    if param_name in shared_params:
                        # print(param_name)
                        weights = attrgetter(param_name)(shared_encoder)
                        # set the shared param to the new model's param
                        param = weights

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
            taskmodels_dict[task_name] = model

        return cls(
            encoder=shared_encoder,
            shared_params=shared_params,
            taskmodels_dict=taskmodels_dict,
        )

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

    def print_parameter_info(self):
        print("Shared Parameters:")
        print("==================")
        shared_param_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name in self.shared_params:
                shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nNon-Shared Parameters:")
        print("======================")
        non_shared_param_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name not in self.shared_params:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )
        for param_name, param in self.named_parameters():
            if param_name not in self.shared_params and "heads" in param_name:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nParameter Counts:")
        print("=================")
        print(f"Shared Parameters: {shared_param_count}")
        print(f"Non-Shared Parameters: {non_shared_param_count}")
        print("Total Parameters: ", shared_param_count + non_shared_param_count)


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
