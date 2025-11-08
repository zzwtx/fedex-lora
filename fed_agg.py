import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch.nn as nn


def aggregate_models_normal(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_ffa(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_ours(global_model, client_models, args):

    global_model = (
        global_model.to("cuda") if torch.cuda.is_available() else global_model
    )
    global_dict = global_model.state_dict()

    for k in global_dict.keys():

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    for client_model in client_models:

        for k in global_dict.keys():

            if "classifier" in k:
                client_model.state_dict()[k].copy_(global_dict[k])

    for name, module in global_model.named_modules():

        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [client_model.state_dict()[lora_A_keys].detach() for client_model in client_models]
            )
            lora_B_weights = torch.stack(
                [client_model.state_dict()[lora_B_keys].detach() for client_model in client_models]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_models))
            ) / len(client_models)

            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.lora_r)
                if args.rslora
                else args.lora_alpha / args.lora_r
            )

            residue = M - lora_B_avg @ lora_A_avg

            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += torch.transpose(
                residue * scaling_factor, 1, 0
            )

    global_model.load_state_dict(global_dict)

    return global_model
