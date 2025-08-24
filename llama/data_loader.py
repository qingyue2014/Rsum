import json
from torch.utils.data import DataLoader, Dataset
import random
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from utils.evaluation import compute_f1, compute_f1_sentence
import copy
import math
from tqdm import tqdm
import torch

speaker_dict = {"Speaker 1": "User", "Speaker 2": "System"}
carecall_dict = {"user":"User", "system":"System"}
NOPERSONA = '__NO__PERSONA__BEAM__MIN__LEN__20__'
DUMMY_TEXT = '__SILENCE__'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)
class SumDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, split='train'):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.split = split

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"

        data_dict = {}
        sample = self.data[index]
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\nYou are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the system. You will receive a previous memory and a dialogue context. Your goal is to update the memory by incorporating the new personality information while ensuring that the memory does not exceed 20 sentences."
        input_context = f"{INPUT_KEY}\n[previous memory]:{sample['init_personas']}\n[dialogue context]:{sample['context']}"
        response = f"{RESPONSE_KEY}\n{sample['labels']}"
        end = f"{END_KEY}"
    
        train_parts = [part for part in [blurb, instruction, input_context, response, end] if part]
        inference_parts = [part for part in [blurb, instruction, input_context] if part]
    
        if self.split == 'train':
            data_dict["input_text"] = "\n\n".join(train_parts)
            data_dict["labels"] = data_dict["input_text"]
        else:
            data_dict["input_text"] = "\n\n".join(inference_parts)
            data_dict["labels"] = sample['labels']
            #print("*****len:",len(data_dict["input_text"].split()))

        return data_dict

    def __len__(self):
        return len(self.data)
    
class MSCDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args, data):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        prompt = [{"role": "system", 
                   "content": "You are an advanced AI language model capable of engaging in personality-based conversations. Respond to the user based on the provided dialogue context. Craft a response that is natural and conversational, while staying within the word limit of 30"}]
        prompt = prompt + item_info[self.args.mode]
        label = item_info["label"]
        dial_id = item_info["dial_id"]
        return {"input": prompt, "label": label, "dial_id": dial_id}
    
    def __len__(self):
        return len(self.data)
    
class NerCollate:
    def __init__(self, args, tokenizer):
        self.instruct_column = args.instruct_column
        self.query_column = args.query_column
        self.response_column = args.response_column
        self.history_column = None
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

    def collate_fn(self, batch):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for example in batch:
            instruction = example["self.instruct_column"]
            input = example[self.query_column]
            output = example[self.response_column]
            instruction = instruction + '\n' + input
            source = prompt.format_map({'instruction': instruction})
            target = f"{self.tokenizer.bos_token}{output}{self.tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = self.tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = (s + t)[:self.max_seq_length]
            labels = ([IGNORE_INDEX] * len(s) + t)[:self.max_seq_length]
            assert len(input_ids) == len(labels)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            labels = labels + [IGNORE_INDEX] * (self.max_seq_length - len(labels))
            # print(input_ids)
            # print(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

            # print(self.tokenizer.decode(input_ids))
            # print(labels)
        results = {'input_ids': torch.tensor(all_input_ids), 'labels': torch.tensor(all_labels)}
        return results

def read_care_data(args, path_name):
    raw_data = json.load(open(path_name))
    data = [[] for i in range(5)]
    for session in raw_data:
        prev_list = []
        guid_id = session[0]["guid"].split("-")[1]
        for sidx, dial in enumerate(session):
            curr_list = []
            for tidx, turn in enumerate(dial["dialogue"]):
                if turn["role"] == "system":
                    if tidx not in [0, 2]:
                        prev_summary = ""
                        if dial["memory"] != []:
                            prev_summary = "User: " + ".".join(dial["memory"])
                        dial_item = {
                            "prev_list": prev_list.copy(),
                            "dial_id": guid_id,
                            "label": turn["text"],
                            "last_turn": tidx == len(dial["dialogue"]) - 1,
                            "full": " ".join(prev_list.copy()) + " " + " ".join(curr_list),
                            "window": " ".join(curr_list),
                            "prev_summary": prev_summary,
                            "summary": ".".join(dial["summary"])
                        }
                        data[sidx].append(dial_item)
                text = carecall_dict[turn["role"]] + ": " + turn["text"]
                curr_list.append(text)

            prev_list.append(" ".join(curr_list))

    return data_train, data_dev, data_test

def count_data(path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    count = 0
    token_num = 0
    for dial in raw_data:
        history = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    history.append("User: " + item["text"])
                else:
                    history.append("System: " + item["text"])
        for idx, item in enumerate(dial["dialog"]):
            if item["id"] == "Speaker 2":
                token_num += len(" ".join(history).split(" "))
                count += 1
            speaker_id, speaker_text = item["id"], item["text"]
            history.append(f"{speaker_id}: {speaker_text}")

    return token_num / count * 1.00

def prepare_test_data(args):
    if args.dataset == "msc":
        path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
        data_test = read_msc_data(args, path_test)
        test_dataset = MSCDataset(args, data_test)
    elif args.dataset == "carecall":
        path_test = f'data/carecall/carecall-memory_en_auto_translated.json'
        data_test = read_care_data(args, path_test)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=test_collate_fn)
                             #sampler = DistributedSampler(test_dataset, shuffle=False), generator=torch.Generator(device="cuda"))
    return data_test, test_loader

def read_msc_data(args, path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    data = []

    for dial in raw_data:
        prev_content = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    prev_content.append({"role":"user","content":item["text"]})
                else:
                    prev_content.append({"role":"assistant","content":item["text"]})
                    
            if prev_content[-1]["role"] == "user":
                prev_content.pop()

        win_content = []
        for idx, item in enumerate(dial["dialog"]):
            speaker_text = item["text"]
            if idx % 2 == 0:
                speaker_role = "user"
            else:
                speaker_role = "assistant"
                #continue
                dial_item = {
                        "full": copy.deepcopy(prev_content + win_content),
                        "window": copy.deepcopy(win_content),
                        "dial_id": item["convai2_id"] + "-" + str(idx),
                        "label": item["text"],
                    }
                data.append(dial_item)

            win_content.append({"role": speaker_role, "content":speaker_text})

    print("#Total sample:", len(data))
    
    return data

def test_collate_fn(batch):
    inputs = [item["input"] for item in batch]
    labels = [item["label"] for item in batch]
    dial_ids = [item["dial_id"] for item in batch]

    return {"input": inputs, "label": labels, "dial_ids":dial_ids}

def prepare_data(args, tokenizer):
    if args.dataset == "msc":
        path_train = f'data/msc_dialog/session_{args.session_id}/train.txt'
        path_dev = f'data/msc_dialog/session_{args.session_id}/valid.txt'
        path_test = f'data/msc_dialog/session_{args.session_id}/test.txt'

        data_train = read_msc_data(args, path_train)
        data_dev = read_msc_data(args, path_dev)
        data_test = read_msc_data(args, path_test)

    elif args.dataset == "carecall":
        path_train = f'data/carecall/carecall-memory_en_auto_translated.json'
        data_train, data_dev, data_test = read_care_data(args, path_test)

    train_dataset = MSCDataset(data_train)
    dev_dataset = MSCDataset(data_dev)
    test_dataset = MSCDataset(data_test)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))

    return train_loader, dev_loader, test_loader




def prepare_sum_data(args):

    path_train = f'data/msc_personasummary/session_{args.session_id}/train.txt'
    path_dev = f'data/msc_personasummary/session_{args.session_id}/valid.txt'
    path_test = f'data/msc_personasummary/session_{args.session_id}/test.txt'

    data_train = read_sum_data(args, path_train, "train")
    data_dev = read_sum_data(args, path_dev, "dev")
    data_test = read_sum_data(args, path_test, "test")

    train_dataset = SumDataset(data_train)
    dev_dataset = SumDataset(data_dev, split='dev')
    test_dataset = SumDataset(data_test, split='test')

    return train_dataset, dev_dataset, test_dataset

def read_sum_data(args, path_name, data_type):

    min_tokens = 100000
    max_tokens = 0
    sum_tokens = 0
    count_data = 0
    def merge_personas(init_personas, add_personas):
        new_personas = [] 
        if add_personas[0] == [] and add_personas[1] == []:
            return init_personas

        for i in range(2):
            personas_i = init_personas[i].copy()
            for ip in add_personas[i]:
                found = False
                for j, op in enumerate(personas_i):
                    if compute_f1_sentence(ip, op) > 0.5:
                        personas_i[j] = ' '.join([op, ip])
                        found = True
                        break
                if not found:
                    personas_i.append(ip)

            new_personas.append(personas_i.copy())
        return new_personas
    

    def get_person_string(personas_list):
        return 'partner\'s persona: '+ ' '.join(personas_list[0]) + ' your persona: '+ " ".join(personas_list[1])


    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    negative_data = []
    #raw_data = random.sample(raw_data, 10)
    for dialog_dict in tqdm(raw_data):
        current_episode = dialog_dict['dialog']
        init_personachat = dialog_dict['init_personachat']
        init_personas = init_personachat['init_personas']
        init_persona_string = get_person_string(init_personas)
        for end_idx in range(len(current_episode)):
            if args.summary_num_turns > 0:
                start_index = max(0, end_idx - args.summary_num_turns + 1)
            else:
                start_index = 0

            add_personas = [[],[]]
            end_line_persona = (
                current_episode[end_idx]['persona_text']
                if 'persona_text' in current_episode[end_idx]
                else NOPERSONA
                )
            
            idx1, idx2 = end_idx, end_idx - 1
            r1, r2 = idx1 % 2 , idx2 % 2 
            add_personas[r1] = current_episode[idx1]['agg_persona_list']
            if idx2 >= 0 :
                add_personas[r2] = current_episode[idx2]['agg_persona_list']

            end_personas = merge_personas(init_personas, add_personas)
            end_persona_string = get_person_string(end_personas)

            dialog_texts = []
            for i in range(start_index, end_idx + 1):
                prefix = "partner: " if i % 2 == 0 else "you: "
                dialog_texts.append(f'{prefix}{current_episode[i]["text"]}')

            action = {
                'context': '\n'.join(dialog_texts),
                'labels': end_persona_string,
                'initial_data_id': dialog_dict['initial_data_id'],
                'init_personas': init_persona_string,
                'utt_idx': end_idx,
                'speaker_idx': end_idx % 2,
                'session_id': args.session_id,
            }
            length = len('\n'.join(dialog_texts).split(" ")) + len(init_persona_string.split(" "))
            if length > max_tokens:
                max_tokens = length
            if length < min_tokens:
                min_tokens = length
            sum_tokens = sum_tokens + length
            count_data = count_data + 1
            if end_line_persona == NOPERSONA:
                negative_data.append(action)
            else:
                data.append(action)

    sum_tokens = sum_tokens / count_data * 1.0

    if data_type == 'train':
        size_to_sample = math.ceil(args.nopersona_subsampling_weight * len(negative_data))
        data.extend(random.sample(negative_data, size_to_sample))
    else:
        data.extend(negative_data)
    
    print(f"#Total number:{len(data)}")
    print(f"{min_tokens}-{max_tokens}:AVE{sum_tokens}")
    return data
