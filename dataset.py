from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import json
import random
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, Union, List
import copy
predefined_prompts = json.load(open("prompt.json"))

PROMPT_TEMPLATE = (
    "{instruction} Assistant: "
)


IGNORE_INDEX = -100

class RSumDataset(Dataset):
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args
        self.prompts = json.load(open("data/msc_dialogue/prompts.json", "r"))
        self.prompt = "You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided. The response is in the form of text and cannot contain emoticons or special characters.The following is the case you need to test:\n The test memory is:{prev_memory}\nThe test dialogue context is:{dialog}\nSo the response to the user is: Assistant:"
        self.example_prompt = self.prompts["gpt-3.5-turbo"]["example1"]
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args.operation == 'infer':
            item_info["input"] = self.prompts['gpt-3.5-turbo']['gen_response_with_memory'].format_map(
                {"prev_memory": item_info["pred_prev_summary"], "dialog": item_info['window']})
        elif self.args.operation == 'ict':
            item_info["input"] = self.prompts['gpt-3.5-turbo']['gen_response_with_memory_example'].format_map(
                {"prev_memory": item_info["pred_prev_summary"], "dialog": item_info['window'], 'example': self.prompts['gpt-3.5-turbo']["example2"]})
        item_info["label"] = item_info["response"]
        return item_info
    
    def __len__(self):
        return len(self.data)
    
class SumDataset(Dataset):
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args
        self.prompts = json.load(open("data/msc_dialogue/prompts.json", "r"))
        self.prompt = "You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided. The response is in the form of text and cannot contain emoticons or special characters.The following is the case you need to test:\n The test memory is:{prev_memory}\nThe test dialogue context is:{dialog}\nSo the response to the user is: Assistant:"
        self.example_prompt = predefined_prompts["gpt-3.5-turbo"]["example"]
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        item_info["input"] = item_info["history"]
        item_info["label"] = item_info["gt_prev_summary"]
        return item_info
    
    def __len__(self):
        return len(self.data)


class RAGDataset(Dataset):
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args
        self.prompt = "You are an advanced AI designed for engaging in natural, personality-based conversations. You possess the ability to remember past interactions, and personal preferences. You will be provided with dialogue context, as well as the relevant historical context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant and aim to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided. The response is in the form of text and cannot contain emoticons or special characters. The following is the history context:{retrieved}\nThe following is the dialogue context: {dialog}\nHere is your response: Assistant:"

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        item_info["input"] = item_info["window"]
        item_info["label"] = item_info["response"]
        return item_info

    def __len__(self):
        return len(self.data)
    
class MSCDataset(Dataset):
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args
        if self.args.dataset == 'msc':
            #self.prompt = "You are an advanced AI designed for engaging in natural, personality-based conversations. You possess the ability to remember past interactions, and personal preferences. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant and aim to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided. The response is in the form of text and cannot contain emoticons or special characters. The following is the dialogue context:\n{dialog}\nHere is your response: Assistant:"
            self.prompt = "You are an advanced AI designed for engaging in natural, personality-based conversations. Your goal is to provide engaging and coherent responses based on the dialogue context provided. The response is in the form of text and cannot contain emoticons or special characters. The following is the dialogue context: {dialog}\nSo the response to the user is: Assistant:"
        elif self.args.dataset == 'carecall':
            self.prompt = "Now, you will play the role of the personal health assistant responsible for monitoring the health status of the user. You possess memory, emotions, and preferences. You should: (1) provide warm companionship to the chatting user; (2) understand past dialogue context and extract information from them to answer questions if they are relevant to the current issue; (3) be an excellent healthy assistant, offering warm and helpful suggestions when users confide their difficulties and seek help. The following is a multi-round conversation between you (the assistant) and the user. Human questions are prefixed with 'User:', while your answers are prefixed with 'Assistant:'. You should refer to the dialogue context, and answer user questions naturally and conversationally. The response is in the form of text and cannot contain emoticons or special characters. The following is the dialogue context:\n{dialog}\nHere is your response: Assistant:"
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        item_info["input"] = self.prompt.format_map({'dialog': item_info[self.args.mode]})
        item_info["label"] = item_info["response"]
        return item_info

    def __len__(self):
        return len(self.data)


class NerCollate:
    def __init__(self, args, tokenizer=None):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = self.args.max_seq_length
        self.max_tokens = 0
        self.total_tokens = 0

    def padding_pairs(self, sources, targets):
        tokenized_sources = self.tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=True)
        all_input_ids = []
        all_labels = []
        length_list = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            if len(s) > self.max_tokens:
                self.max_tokens = len(s)
            self.total_tokens = self.total_tokens + len(s)
            length_list.append(len(s)+len(t))
        max_length = max(length_list)
        if max_length > self.max_seq_length:
            max_length = self.max_seq_length
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = (s + t)[:max_length]
            labels = ([IGNORE_INDEX] * len(s) + t)[:max_length]
            assert len(input_ids) == len(labels)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.tensor(all_input_ids)
        labels = torch.tensor(all_labels)
        return {'input_ids': input_ids, 'labels': labels}
    
    def padding_inputs(self, sources):
        results = self.tokenizer.batch_encode_plus(sources, return_tensors="pt", padding=True)
        return results

    def collate_fn(self, batch):
        sources = []
        targets = []
        dial_ids = []
        prompt = PROMPT_TEMPLATE
        for example in batch:
            input = example["input"]
            output = example["label"]
            source = input
            dial_ids.append(example['dial_id'])

            sources.append(source)
            targets.append(output)

        if self.args.operation == 'train':
            return self.padding_pairs(sources, targets)
        else:
            if 'gpt' in self.args.model_name:
                return {"input": sources, "label": targets, "dial_ids":dial_ids}
            else:
                return self.padding_inputs(sources)
