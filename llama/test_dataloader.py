import json
from torch.utils.data import DataLoader, Dataset
import random
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from utils.evaluation import compute_f1, compute_f1_sentence
#from utils.rag import rag
import copy
import math
from tqdm import tqdm
import torch
import os
count = 0

random.seed(42)

speaker_dict = {"Speaker 1": "User", "Speaker 2": "System"}
carecall_dict = {"user":"User", "system":"System"}
NOPERSONA = '__NO__PERSONA__BEAM__MIN__LEN__20__'
DUMMY_TEXT = '__SILENCE__'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
PROMPT_TEMPLATE = {
    "summarize":"<s>[INST]<<SYS>>\nYou are a helpful assistent\n<</SYS>>\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n Based on the previous text, provide a updated single summary:[/INST]",
    "rsum_dial":"[INST]<<SYS>>\n{instruction}\n<</SYS>>\n\n{input}[/INST]"
}
IGNORE_INDEX=-100
class SumDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, local_rank, split='train'):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.split = split
        self.local_rank = local_rank

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data_dict = {}
        sample = self.data[index]
        data_dict["instruction"] = "You are an advanced AI language model with the ability to generate a summary to keep track of key personality information for both of speakers. You will receive a dialogue context and previous summary. Your goal is to generate new summary by incorporating the dialogue context into previous memory and ensuring that the new summary does not exceed 20 sentences. To successfully generate the summary, follow these steps: 1. Carefully extract the key personality information of the two speakers. 2. Consider the dialogue context provided to identify any personality traits. 3. Pay attention to the relevance and importance of the personality information, focusing on capturing the most significant aspects while maintaining the overall coherence of the summary. Remember, the summary should serve as a reference point to maintain continuity in the dialogue and help you respond accurately to the user based on the personality traits of two speakers.The updated summary should be in the following format: {\"user persona\": \"Description of the user's person\", \"assistant persona\": \"Description of the assistant's persona\"}."
        data_dict["input"] = f"Dialogue: '{sample['context']}'\nPrevious summary: {sample['prev_summary']}\n"
        data_dict["prev_summary"] = sample['prev_summary']
        data_dict["label"] = sample['label']
        data_dict["dial_id"] = sample['dial_id']

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

class RSumDataset(Dataset):
    def __init__(self, data, local_rank, split='train'):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.split = split
        self.local_rank = local_rank

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data_dict = {}
        sample = self.data[index]
        data_dict["instruction"] = "You are an advanced AI language model capable of engaging in personality-based conversations. You will be provided with personality information of both speakers and a dialogue context. Your goal is to craft a response that is natural and conversational, while staying within the word limit of 30."
        #data_dict["input"] = f"Previous summary: {sample['prev_summary']}\nDialogue: '{sample['context']}\n"
        data_dict["input"] = f"Personas: {sample['prev_summary']}\nDialogue: {sample['context']}\n"
        data_dict["prev_summary"] = sample['prev_summary']
        data_dict["label"] = sample['label']
        data_dict["dial_id"] = sample['dial_id']

        return data_dict

    def __len__(self):
        return len(self.data)
    
class NerCollate:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length

    def collate_fn(self, batch):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE[self.args.mode]
        for example in batch:
            instruction = example["instruction"]
            input = example["input"]
            output = example["label"]
            source = prompt.format_map({'instruction': instruction,"input":input})
            target = f"{self.tokenizer.bos_token}{output}{self.tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = self.tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        length_list = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
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
        results = {'input_ids': torch.tensor(all_input_ids), 'labels': torch.tensor(all_labels)}
        return results

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

def collate_fn(batch):
    inputs = [item["input"] for item in batch]
    labels = [item["label"] for item in batch]
    dial_ids = [item["dial_id"] for item in batch]
    return {"input": inputs, "label": labels, "dial_ids":dial_ids}

def prepare_test_data(args):
    if args.dataset == "msc":
        path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
        data_test = read_msc_data(args, path_test)
        test_dataset = MSCDataset(args, data_test)
    elif args.dataset == "carecall":
        path_test = f'data/carecall/carecall-memory_en_auto_translated.json'
        data_test = read_care_data(args, path_test)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    return data_test, test_loader

def read_msc_data(args, path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    data = []

    raw_data = random.sample(raw_data, 20)
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


def sum_collate_fn(batch, args):
    inputs = []
    labels = []
    dial_ids = []
    prev_sums = []
    prompt = PROMPT_TEMPLATE[args.operation]
    for example in batch:
        instruction = example["instruction"]
        input = example["input"]
        output = example["label"]
        prev_sums.append(example["prev_summary"])
        source = prompt.format_map({'instruction': instruction,"input":input})
        inputs.append(source)
        labels.append(output)
        dial_ids.append(example["dial_id"])

    return {"input": inputs, "label": labels, "dial_ids":dial_ids, "prev_summary": prev_sums}

def prepare_rsum_data(args):
    path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
    data_test = read_rsum_data(args, path_test, "test")
    test_dataset = RSumDataset(data_test, args.local_rank, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, 
                             shuffle=False, collate_fn=partial(sum_collate_fn, args=args), num_workers=0)
    return test_loader

def read_rsum_data(args, path_name, dtype):
    def load_summary_file(args):
        file_path = os.path.join(args.saving_dir, f"{args.dataset}_summarize_sid{args.session_id}.json")
        with open(file_path, 'r') as f:
            result = json.load(f)
            for key in result.keys():
                #result[key]["prediction"] = "".join(result[key]["prediction"].split(":")[1:])
                s1 = result[key]["prediction"].split("{")[1]
                result[key]["prediction"] = s1.split("}")[0]
        return result

    def get_person_string(personas_list):
        return 'User\'s persona: '+ ' '.join(personas_list[0]) + ' Assistant persona: '+ " ".join(personas_list[1])

    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    persona_list = {}
    prev_summary_set = load_summary_file(args)
    raw_data = random.sample(raw_data, 20)
    for dialog_dict in tqdm(raw_data):
        init_personas = dialog_dict['init_personas']
        init_persona_string = get_person_string(init_personas)
        dial_id = dialog_dict["metadata"]["initial_data_id"]
        prev_summary = "None"
        if dial_id in prev_summary_set.keys():
            prev_summary = prev_summary_set[dial_id]["prediction"]

        dialog_texts = []
        prev_dialogs = dialog_dict['previous_dialogs'][-1]["dialog"]
        for i in range(0, len(prev_dialogs)):
            prefix = "User: " if i % 2 == 0 else "Assistant: "
            dialog_texts.append(f'{prefix}{prev_dialogs[i]["text"]}')

        
        for tidx, turn in enumerate(dialog_dict["dialog"]):
            prefix = "User: " if tidx % 2 == 0 else "Assistant: "
            if prefix == "Assistant: ":
                action = {
                    'context': '\n'.join(dialog_texts),
                    'label': turn["text"],
                    'init_personas': init_persona_string,
                    'personas':dialog_dict["personas"],
                    'prev_summary': prev_summary,
                    'dial_id': dial_id+"-" + str(tidx),
                }
                data.append(action)
            dialog_texts.append(f'{prefix}{turn["text"]}')
        persona_list[dial_id] = dialog_dict["personas"][1]

    persona_file = os.path.join("save/personas", f"msc_persona_sid{args.session_id}.json")
    with open(persona_file,"w", encoding='utf-8') as f: 
        f.write(json.dumps(persona_list, ensure_ascii=False, indent=4))  

    return data
   
def prepare_test_sum_data(args):
    path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
    data_test = read_test_sum_data(args, path_test, "test")
    test_dataset = SumDataset(data_test, args.local_rank, split='test')
    #sampler = DistributedSampler(test_dataset, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=sum_collate_fn, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, 
                             shuffle=False, collate_fn=partial(sum_collate_fn, args=args), num_workers=0)
    return test_loader

def read_test_sum_data(args, path_name, dtype):
    def merge_personas(init_personas, add_personas):
        new_personas = [] 
        if add_personas[0] == [] and add_personas[1] == []:
            return init_personas

        for i in range(2):
            personas_i = init_personas[i].copy()
            for ip in add_personas[i]:
                found = False
                for j, op in enumerate(personas_i):
                    if compute_f1_sentence(ip, op) > 0.4:
                        personas_i[j] = ' '.join([op, ip])
                        found = True
                        break
                if not found:
                    personas_i.append(ip)

            new_personas.append(personas_i.copy())
        return new_personas
    def get_person_string(personas_list):
        return 'User\'s persona: '+ ' '.join(personas_list[0]) + ' Assistant persona: '+ " ".join(personas_list[1])
    
    def load_summary_file(args):
        if args.summary_file == "":
            return None
        file_path = os.path.join(args.saving_dir, args.summary_file)
        with open(file_path, 'r') as f:
            result = json.load(f)
            for key in result.keys():
                result[key]["prediction"] = "".join(result[key]["prediction"].split(":")[1:])
        return result

    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    prev_summary_set = load_summary_file(args)
    raw_data = random.sample(raw_data, 20)
    for dialog_dict in tqdm(raw_data):
        prev_dialogs = dialog_dict['previous_dialogs'][-1]["dialog"]
        init_personas = dialog_dict['init_personas']
        init_persona_string = get_person_string(init_personas)
        dialog_texts = []
        for i in range(0, len(prev_dialogs)):
            prefix = "User: " if i % 2 == 0 else "Assistant: "
            dialog_texts.append(f'{prefix}{prev_dialogs[i]["text"]}')

        add_personas = dialog_dict["personas"]
        end_personas = merge_personas(init_personas, add_personas)
        end_persona_string = get_person_string(end_personas)
        dial_id = dialog_dict["metadata"]["initial_data_id"]
        prev_summary = "None"
        if prev_summary_set is not None and dial_id in prev_summary_set.keys():
            prev_summary = prev_summary_set[dial_id]["prediction"]
        action = {
                'context': '\n'.join(dialog_texts),
                'label': end_persona_string,
                'init_personas': init_persona_string,
                'prev_summary': prev_summary,
                'dial_id': dial_id,
            }
        data.append(action)
    return data

def read_sum_data(args, path_name, data_type):
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
    raw_data = random.sample(raw_data, 20)
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
                'context': ' '.join(dialog_texts),
                'labels': end_persona_string,
                'initial_data_id': dialog_dict['initial_data_id'],
                'init_personas': init_persona_string,
                'utt_idx': end_idx,
                'speaker_idx': end_idx % 2,
                'session_id': args.session_id,
            }
            if end_line_persona == NOPERSONA:
                negative_data.append(action)
            else:
                data.append(action)

    if data_type == 'train':
        size_to_sample = math.ceil(args.nopersona_subsampling_weight * len(negative_data))
        data.extend(random.sample(negative_data, size_to_sample))
    else:
        data.extend(negative_data)
    
    print(f"#Total number:{len(data)}")
    return data
