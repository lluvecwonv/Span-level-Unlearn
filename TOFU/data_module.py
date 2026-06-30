import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
import numpy as np
from utils import get_model_identifiers_from_yaml


def convert_chat_template_to_model_format(tokenizer, max_length, question, answer, model_configs):
    """
    Use tokenizer.apply_chat_template() for models like Qwen, Mistral, etc.
    These models have their own chat format and EOS tokens.

    EOS tokens by model:
    - Qwen2.5: <|im_end|> (ID: 151645)
    - Mistral: </s> (ID: 2)
    """
    # Build chat messages
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

    # Apply chat template - this automatically adds proper formatting and EOS
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    # Calculate question token length for label masking
    question_messages = [{"role": "user", "content": question}]
    question_text = tokenizer.apply_chat_template(
        question_messages,
        tokenize=False,
        add_generation_prompt=True  # Include the assistant prompt start
    )
    num_question_tokens = len(tokenizer.tokenize(question_text, add_special_tokens=True))

    # Padding
    pad_length = max_length - len(encoded.input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_input_ids = encoded['input_ids'] + [pad_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [-100] * pad_length

    # Mask question tokens in label
    for i in range(min(num_question_tokens, len(label))):
        label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    # Add EOS token explicitly to the answer so model learns to generate it
    new_answer = answer_token + answer + tokenizer.eos_token
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [-100] * pad_length

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


def convert_answer_only_to_model_format(tokenizer, max_length, answer, model_configs, model_family=None):
    """
    Answer-only version that removes the question completely.
    This keeps activation/gradient covariance consistent for influence computation.
    """
    # Llama models need the answer-start marker.
    if model_family and 'llama' in model_family.lower():
        # For Llama, question_end_tag marks the start of the answer.
        answer_start_token = model_configs.get('question_end_tag', '')
        # Add EOS token explicitly so model learns to generate it
        answer_text = answer_start_token + answer + tokenizer.eos_token
    else:
        # Other models use the existing answer_tag format.
        answer_token = model_configs.get('answer_tag', '')
        answer_text = answer_token + answer + tokenizer.eos_token

    encoded = tokenizer(
        answer_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [-100] * pad_length

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)



class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk", use_tif=False, uw_identifier=None):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else: # load dataset from local files.
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        # Check if model uses chat template (Qwen, Mistral, etc.)
        self.use_chat_template = self.model_configs.get('use_chat_template', 'false').lower() == 'true'

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

        # TIF: Unwanted Word identification
        self.use_tif = use_tif
        self.uw_identifier = uw_identifier
        self.uw_masks_cache = {}  # Cache UW masks by index

        if self.use_tif and self.uw_identifier:
            total_samples = len(self.forget_data)
            print(f"[TIF] Precomputing UW masks for {total_samples} forget samples...")
            print(f"[TIF] Progress: ", end='', flush=True)

            for idx in range(total_samples):
                # Show progress every 10% or every sample if less than 10 samples
                if total_samples < 10 or idx % max(1, total_samples // 10) == 0:
                    percent = int(100 * idx / total_samples)
                    print(f"{percent}%...", end='', flush=True)

                question = self.forget_data[idx]['question']
                answer = self.forget_data[idx]['answer']
                try:
                    uw_mask = self.uw_identifier.identify(question, answer, self.tokenizer)
                    self.uw_masks_cache[idx] = uw_mask
                except Exception as e:
                    print(f"\n[TIF] Warning: Failed to identify UW for sample {idx}: {e}")
                    # Fallback: all tokens are UW
                    answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                    self.uw_masks_cache[idx] = torch.ones(len(answer_tokens), dtype=torch.bool)

            print(f" 100% Done!")
            print(f"[TIF] Completed UW mask precomputation for {total_samples} samples")

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data

            torch.manual_seed(idx)
            idx_actual = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx_actual]['question']
            answer = data[idx_actual]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            if self.use_chat_template:
                converted_data = convert_chat_template_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            else:
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)

            # TIF: Add UW mask for forget samples
            if self.use_tif and data_type in ["forget", "idk"] and idx in self.uw_masks_cache:
                converted_data['uw_mask'] = self.uw_masks_cache[idx]

            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        # Check if model uses chat template (Qwen, Mistral, etc.)
        self.use_chat_template = self.model_configs.get('use_chat_template', 'false').lower() == 'true'

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            question = data[idx]['question']

            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            if self.use_chat_template:
                converted_data = convert_chat_template_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            else:
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetKTOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetKTOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        # Check if model uses chat template (Qwen, Mistral, etc.)
        self.use_chat_template = self.model_configs.get('use_chat_template', 'false').lower() == 'true'

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)

            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            question = data[idx]['question']

            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            if self.use_chat_template:
                converted_data = convert_chat_template_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            else:
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer', answer_only=False, max_num=-1):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_only = answer_only  # Whether to use only the answer.
        self.model_family = model_family  # Store model family for Llama checks.
        # Load dataset
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        # Check if model uses chat template (Qwen, Mistral, etc.)
        self.use_chat_template = self.model_configs.get('use_chat_template', 'false').lower() == 'true'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            if self.answer_only:
                # Use only the answer and remove the question.
                converted_data = convert_answer_only_to_model_format(
                    self.tokenizer, self.max_length, answer, self.model_configs, self.model_family
                )
            elif self.use_chat_template:
                # Models that use chat templates, such as Qwen and Mistral.
                converted_data = convert_chat_template_to_model_format(
                    self.tokenizer, self.max_length, question, answer, self.model_configs
                )
            else:
                # Existing format for models such as Llama 2 and Llama 3.
                converted_data = convert_raw_data_to_model_format(
                    self.tokenizer, self.max_length, question, answer, self.model_configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

class TNPOForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TNPOForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.forget_mask = None
        # Check if model uses chat template (Qwen, Mistral, etc.)
        self.use_chat_template = self.model_configs.get('use_chat_template', 'false').lower() == 'true'

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            with open(self.idontknowfile, "r") as f:
                self.idk = f.readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def set_forget_mask(self, forget_mask):
        """Set the forget mask."""
        self.forget_mask = forget_mask
        print(f"Forget mask set with {len(forget_mask)} entries")
        if len(forget_mask) > 0:
            print(f"First mask shape: {forget_mask[0].shape}, dtype: {forget_mask[0].dtype}")
            print(f"First mask sum: {forget_mask[0].sum()}")

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            data_idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[data_idx]['question']
            answer = data[data_idx]['answer']

            if data_type == "idk":
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            if self.use_chat_template:
                converted_data = convert_chat_template_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            else:
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            
            if data_type == "forget" and self.forget_mask is not None and idx < len(self.forget_mask):
                mask = self.forget_mask[idx]
                
                if len(mask) < len(converted_data[0]):
                    adjusted_mask = torch.cat([
                        mask, 
                        torch.zeros(len(converted_data[0]) - len(mask), dtype=torch.bool)
                    ])
                else:
                    adjusted_mask = mask[:len(converted_data[0])]
                
                converted_data = converted_data + (adjusted_mask,)
            else:
                empty_mask = torch.zeros(len(converted_data[0]), dtype=torch.bool)
                converted_data = converted_data + (empty_mask,)
            
            rets.append(converted_data)
        return rets

import random
class UnlearnDataset(Dataset):
    def __init__(self, datasets):
        self.forget_dataset = datasets.get("forget", None)
        self.retain_dataset = datasets.get("retain", None)

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        if self.retain_dataset:
            return len(self.retain_dataset)
        raise ValueError("No dataset available.")

    def __getitem__(self, idx):
        forget_data = self.forget_dataset[idx]
        retain_idx = random.randint(0, len(self.retain_dataset) - 1)
        retain_data = self.retain_dataset[retain_idx]
        data = [forget_data, retain_data]
        return data

def unlearncollector(samples):
    res = {"forget": None, "retain": None}
    if samples["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples])
        )
    if samples["retain"]:
        retain_samples = [sample["retain"] for sample in samples]
        res["retain"] = (
            torch.stack([sample["input_ids"] for sample in retain_samples]),
            torch.stack([sample["attention_mask"] for sample in retain_samples]),
            torch.stack([sample["label"] for sample in retain_samples])
        )
    return res

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    # Include the mask when present.
    if len(samples[0]) > 3:
        forget_mask = [s[3] for s in samples]
        return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(forget_mask)
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    """Data collator that includes dataset indices."""
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]  # Use the original dataset indices.
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
