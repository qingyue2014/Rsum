from chatgpt.data_loader import prepare_data, prepare_test_data
from config import get_args, get_logger
from chatgpt.robot import gpt_summary_results, gpt_response_results, davinci_response_results, davinci_summary_results
from tqdm import tqdm
from utils.evaluation import compute_f1, calc_distinct
import time
import pickle
import os, json

predefined_prompts = json.load(open("prompt.json"))

def update_summary(args, context="", summary="", example=""):
    instruction = predefined_prompts[args.dataset]['gpt-3.5-turbo']["update_memory"]
    if args.do_ict:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Examples** {example}\n " \
                 f"**Test** [Previous Memory] {summary} [Dialogue Context] {context} [Updated Memory]"
    else:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Test** [Previous Memory] {summary} [Dialogue Context] {context} [Updated Memory]"
    if "gpt" in args.model_name:
        result = gpt_summary_results(args, prompt)
    elif "davinci" in args.model_name:
        result = davinci_summary_results(args, prompt)
    result = result.replace("\n", "")
    return result

def update_response(args, summary="", context="", example=""):
    instruction = predefined_prompts[args.dataset]['gpt-3.5-turbo']["update_response"]
    if args.do_ict:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Examples** {example}\n " \
                 f"**Test** [Previous Memory] {summary} [Dialogue Context] {context} [Response] \n"
    else:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Test** [Previous Memory] {summary} [Dialogue Context] {context} [Response] \n"

    if "gpt" in args.model_name:
        result = gpt_response_results(args, prompt)
    elif "davinci" in args.model_name:
        result = davinci_response_results(args, prompt)
    result = result.replace("\n","")
    result = result.replace("System:", "")
    return result

def make_direct_response(args, context, example):
    instruction = predefined_prompts[args.dataset]['gpt-3.5-turbo']["direct_response"]
    if args.do_ict:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Examples** {example}\n " \
                 f"**Test** [Dialogue Context]{context} [Response] "
    else:
        prompt = f"**Instruction** {instruction}\n " \
                 f"**Test** [Dialogue Context] {context} [Response] "
    if "gpt" in args.model_name:
        result = gpt_response_results(args, prompt)
    elif "davinci" in args.model_name:
        result = davinci_response_results(args, prompt)
    result = result.replace("\n","")
    result = result.replace("System:", "")
    return result

def load_prev_summary(args):
    prev_session = args.session_id - 1
    if args.dataset == 'msc':
        with open(os.path.join(f'save_msc/gpt-3.5-old', f"summary_dicts_sid{prev_session}.pkl"), 'rb') as fr:
            summary_dict = pickle.load(fr)
    else:
        with open(os.path.join(f'{args.saving_dir}', f"infer_sum_sid5.json"), 'rb') as fr:
            summary_dict = json.load(fr)
    return summary_dict

def get_prev_summary(args, dial, example):
    prev_summary = "Empty"
    for idx, prev in enumerate(dial["prev_list"]):
        prev_summary = update_summary(args=args, context=prev, summary=prev_summary, example=example)
    return prev_summary

def summary_all(args, prefix="sumall"):
    test_data, example = prepare_test_data(args)
    summary_text = ""
    predictions = []
    references = []
    mem_preds, mem_labels = [], []
    summary_file = open(os.path.join(args.saving_dir, f"{prefix}_summary.txt"), "w")
    args.logger = get_logger('{}/{}.log'.format(args.saving_dir, args.mode), "a")
    args.logger.info(args)
    pred_dicts = {}
    summary_dict = {}
    curr_summary_dicts = {}
    prev_summary_dicts = None
    
    if args.session_id > 2:
        prev_summary_dicts = load_prev_summary(args)

    ex_summ, ex_resp ="", ""
    if args.do_ict:
        ex_summ = example["update_summary"]
        ex_resp = example["update_response"]

    for idx, test_dial in enumerate(tqdm(test_data)):
        cur_dial_id = test_dial["dial_id"]
        init_dial_id = cur_dial_id.split("-")[0]

        if args.dataset == 'msc':
            if test_dial["first_turn"]:
                #import pdb; pdb.set_trace()
                if prev_summary_dicts is None or init_dial_id not in prev_summary_dicts:
                    summary_text = get_prev_summary(args, test_dial, ex_summ)
                else:
                    summary_text = prev_summary_dicts[init_dial_id]
        else:
            if prev_summary_dicts is not None and init_dial_id in prev_summary_dicts:
                summary_text = prev_summary_dicts[init_dial_id]
            else:
                summary_text = get_prev_summary(args, test_dial, ex_summ)
                summary_dict[init_dial_id] = summary_text
            

        response = update_response(args, summary=summary_text, context=test_dial["window"], example=ex_resp)
        predictions.append(response)
        ground_truth = test_dial["label"]
        references.append(ground_truth)
        pred_dicts[cur_dial_id] = {'prediction': response, 'label':ground_truth}

        #the summary will be updated at the last turn.
        if test_dial["last_turn"] and args.session_id != 5:
            summary_text = update_summary(args, context=test_dial["window"], summary=summary_text)
            curr_summary_dicts[cur_dial_id] = summary_text


    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    wfile= os.path.join(args.saving_dir, f"{args.operation}_sum_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(summary_dict, ensure_ascii=False, indent=4))  

    #print_eval_metrics(args, predictions, references, mem_preds, mem_labels)

def summary_gold(args, prefix="sumgold"):
    test_data, example = prepare_test_data(args)
    predictions = []
    references = []
    refs_file = open(os.path.join(args.saving_dir, f"{prefix}_refs.txt"), "w")
    hyps_file = open(os.path.join(args.saving_dir, f"{prefix}_hyps.txt"), "w")
    args.logger = get_logger('{}/{}.log'.format(args.saving_dir, args.mode), "a")
    args.logger.info(args)

    ex_summ, ex_resp ="", ""
    if args.do_ict:
        ex_resp = example["update_response"]

    for idx, test_dial in enumerate(tqdm(test_data)):
        response = update_response(args, summary=test_dial["prev_summary"], context=test_dial["window"],
                                   example=ex_resp)
        predictions.append(response)
        ground_truth = test_dial["label"]
        references.append(ground_truth)
        refs_file.writelines(test_dial["label"] + "\n")
        hyps_file.writelines(response + "\n")
    refs_file.close()
    hyps_file.close()
    print_eval_metrics(args, predictions, references, [], [])

def make_file_name(args):
    prefix = args.mode
    if args.do_ict:
        prefix += "_ict"
    if args.do_sample:
        prefix += "_sample"
    return prefix

def summary_sample(args):
    test_data, example = prepare_test_data(args)
    predictions = []
    references = []
    mem_preds, mem_labels = [], []
    prefix = make_file_name(args)
    refs_file = open(os.path.join(args.saving_dir, f"{args.dataset}_{prefix}_refs.txt"), "a")
    hyps_file = open(os.path.join(args.saving_dir, f"{prefix}_hyps.txt"), "a")
    summary_file = open(os.path.join(args.saving_dir, f"{prefix}_summary.txt"), "a")
    args.logger = get_logger('{}/{}.log'.format(args.saving_dir, args.mode), "w")
    for idx, test_dial in enumerate(tqdm(test_data)):
        if idx <= 49:
            continue
        cur_dial_id = test_dial["dial_id"]
        summary_text = get_prev_summary(args, test_dial, example["update_summary"])
        response = update_response(args, summary=summary_text, context=test_dial["window"],
                                   example=example["update_response"])

        predictions.append(response)
        ground_truth = test_dial["label"]
        references.append(ground_truth)
        refs_file.writelines(test_dial["label"] + "\n")
        hyps_file.writelines(response + "\n")

        memory = test_dial["prev_summary"]
        summary_file.writelines(f"{cur_dial_id}${summary_text}\n")
        summary_file.writelines(f"\n")
        mem_preds.append(summary_text)
        mem_labels.append(memory)

    refs_file.close()
    hyps_file.close()
    summary_file.close()
    print_eval_metrics(args, predictions, references, mem_preds, mem_labels)


def direct_response(args, prefix="direct"):
    test_data, examples = prepare_test_data(args)
    predictions = []
    references = []
    pred_dicts = {}
    args.logger = get_logger('{}/{}.log'.format(args.saving_dir, args.mode), "w")
    args.logger.info("Running!")
    for idx, test_dial in enumerate(tqdm(test_data)):
        dial_id = test_dial['dial_id']
        response = make_direct_response(args, test_dial[args.mode], examples[args.mode])
        pred_dicts[dial_id] = {'prediction': response, 'label': test_dial["label"]}

    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    #print_eval_metrics(args, predictions, references)


def print_eval_metrics(args, predictions, references, mem_preds=[], mem_labels=[]):
    bleu_dict = bleu.compute(predictions=predictions, references=references)["precisions"]
    bleu1, bleu2, bleu3 = round(bleu_dict[0], 4), round(bleu_dict[1], 4), round(bleu_dict[2], 4),
    f1_score = compute_f1(predictions, references)
    mem_score = compute_f1(mem_preds, mem_labels)
    dist_score = calc_distinct(predictions)
    args.logger.info(f"bleu1/2:{bleu1*100}/{bleu2*100}, f1/mem_f1:{f1_score*100}/{mem_score}, "
                     f"dist1/2:{dist_score[0] * 100}/{dist_score[1] * 100}")

if __name__ == "__main__":
    args = get_args()
    if args.dataset == 'msc':
        args.saving_dir = 'save_msc'
    else:
        args.saving_dir = 'save_carecall'
    args.saving_dir = os.path.join(args.saving_dir, args.model_name)
    if not os.path.exists(args.saving_dir):
        os.makedirs(args.saving_dir)
    if args.mode == "rsum":
        if args.do_sample:
            summary_sample(args)
        else:
            summary_all(args)
    elif args.mode == "sumgold":
        summary_gold(args)
    elif args.mode in ["full", "window"]:
        direct_response(args)

