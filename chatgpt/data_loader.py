import json
from torch.utils.data import DataLoader, Dataset
import random
from functools import partial

speaker_dict = {"Speaker 1": "User", "Speaker 2": "System"}
carecall_dict = {"user":"User", "system":"System"}

class MSCDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data):
        """Reads source and target sequences from txt files."""
        self.data = data

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
                    ### Instruction:
                    You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the system. You will receive a memory and a dialogue context. Your goal is to update the memory by incorporating the new personality information while ensuring that the memory does not exceed 20 sentences.
                    ### Input:
                    [Previous memory] {item_info["prev_summary"]} [Dialogue Context] {item_info["window"]}
                    ### Response:
                    {item_info["curr_summary"]}"""
        return prompt

    def __len__(self):
        return len(self.data)

def read_msc_data(args, path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    data = []

    for dial in raw_data:
        prev_list = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            prev["history"] = []
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    prev["history"].append("User: " + item["text"])
                else:
                    prev["history"].append("System: " + item["text"])
            prev_list.append(" ".join(prev["history"]))
        curr_list = []
        for idx, item in enumerate(dial["dialog"]):
            speaker_text = item["text"]
            if idx % 2 == 0:
                speaker_role = "User"
            else:
                speaker_role = "System"
                dial_item = {
                    "prev_list": prev_list,
                    "curr_list": curr_list.copy(),
                    "full": " ".join(prev_list.copy()) + " " + " ".join(curr_list.copy()),
                    "window": " ".join(curr_list.copy()),
                    "dial_id": item["convai2_id"] + '-' + str(idx),
                    "first_turn": int(idx / 2) == 0,
                    "label": item["text"],
                    "last_turn": idx == len(dial["dialog"]) - 1,
                    "personas": dial["personas"]
                }
                data.append(dial_item)
            curr_list.append(f"{speaker_role}: {speaker_text}")
    return data

def read_care_data(args, path_name):
    raw_data = json.load(open(path_name))
    data = [[] for i in range(5)]
    for session in raw_data:
        prev_list = []
        guid_id = session[0]["guid"].split("-")[1]
        for sidx, dial in enumerate(session):
            curr_list = []
            for tidx, turn in enumerate(dial["dialogue"]):
                turn_id = guid_id + '-' + str(tidx)
                if turn["role"] == "system":
                    if tidx not in [0, 2]:
                        prev_summary = ""
                        if dial["memory"] != []:
                            prev_summary = "User: " + ".".join(dial["memory"])
                        dial_item = {
                            "prev_list": prev_list.copy(),
                            "dial_id": turn_id,
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

    random.seed(args.random_seed)
    select_data = data[args.session_id-1][:1000]
    example_data = random.sample(data[args.session_id-1], args.n_shot)
    example = {"window": "", "full": "", "update_response": "", "update_summary": ""}
    for item in example_data:
        window_context, full_context = item["window"], item["full"]
        prev_summary = item["prev_summary"]
        summary = item["summary"]
        response = item["label"]
        example["window"] += f"[Dialogue Context] {window_context} [Response] {response}"
        example[
            "update_summary"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Updated Memory] {summary}"
        example[
            "update_response"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Response] {response}"
        example["full"] += f"[Dialogue Context] {full_context} [Response] {response}"
    return select_data, example

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


def load_example(args, path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    summaries = json.load(open("data/msc_dialogue/sessionlevel_summaries_subsample5.json", "r"))

    data = []

    for dial in raw_data:
        prev_list = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            prev["history"] = []
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    prev["history"].append("User: " + item["text"])
                else:
                    prev["history"].append("System: " + item["text"])
            prev_list.append(" ".join(prev["history"]))
        curr_list = []
        for idx, item in enumerate(dial["dialog"][:-1]):
            speaker_text = item["text"]
            if idx % 2 == 0:
                speaker_role = "User"
            else:
                speaker_role = "System"

            curr_list.append(f"{speaker_role}: {speaker_text}")

        dial_id = dial["metadata"]['initial_data_id']
        session_id = dial["metadata"]["session_id"]
        dial_item = {
            "prev_list": prev_list,
            "curr_list": curr_list,
            "full": " ".join(prev_list) + " " + " ".join(curr_list),
            "window": " ".join(curr_list),
            "dial_id": dial_id,
            "prev_summary": summaries["1"][dial_id],
            "summary": summaries["2"][dial_id],
            "label": dial["dialog"][-1]["text"],
            "personas": dial["personas"]
        }
        # print(" ".join(history[-args.window_size*2:]))
        data.append(dial_item)

    random.seed(args.random_seed)
    data = random.sample(data, args.n_shot)
    example = {"window":"", "full":"", "update_response":"", "update_summary":""}
    for item in data:
        window_context, full_context = item["window"], item["full"]
        prev_summary = "User:" + " ".join(item["prev_summary"][0]) + "System:" + " ".join(item["prev_summary"][1])
        summary = "User:" + " ".join(item["summary"][0]) + "System:" + " ".join(item["summary"][1])
        response = item["label"]
        example["window"] += f"[Dialogue Context] {window_context} [Response] {response}"
        example["update_summary"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Updated Memory] {summary}"
        example["update_response"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Response] {response}"
        example["full"] += f"[Dialogue Context] {full_context} [Response] {response}"
    return example

def prepare_test_data(args):
    if args.dataset == "msc":
        path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
        data_test = read_msc_data(args, path_test)
        path_valid = f'data/msc_dialogue/session_2/valid.txt'
        examples = load_example(args, path_valid)
    elif args.dataset == "carecall":
        path_test = f'data/carecall/carecall-memory_en_auto_translated.json'
        data_test, examples = read_care_data(args, path_test)

    return data_test, examples

def collate_fn(data, tokenizer):
    encoding = tokenizer.batch_encode_plus(
        data,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "labels": encoding["input_ids"].squeeze()
    }
    return batch_data


def prepare_data(args, tokenizer):
    if args.dataset == "msc":
        path_train = f'data/msc_dialogue/session_{args.session_id}/train.txt'
        path_dev = f'data/msc_dialogue/session_{args.session_id}/valid.txt'
        path_test = f'data/msc_dialogue/session_{args.session_id}/test.txt'
    elif args.dataset == "carecall":
        path_train = f'data/carecall/carecall-memory_en_auto_translated.json'

    data_train = read_msc_data(args, path_train)
    data_dev = read_msc_data(args, path_dev)
    data_test = read_msc_data(args, path_test)

    train_dataset = MSCDataset(data_train)
    dev_dataset = MSCDataset(data_dev)
    test_dataset = MSCDataset(data_test)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args))
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args))

    return train_loader, dev_loader, test_loader
