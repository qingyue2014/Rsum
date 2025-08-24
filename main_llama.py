import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from utils.robot import gpt_response_results, gpt_memory_results
from utils.llm_judge import run_llm_judge, load_eval_file, run_llm_win
from tqdm import tqdm
from dataloader import load_dataset
from dataset import NerCollate
from config import get_args
import json
import tiktoken
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainerCallback, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationConfig
from utils.evaluation import compute_f1, evaluate_corpus, evaluate_memory
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.distributed as dist
#from chatgpt.robot import gpt_response_results
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import torch
import numpy as np
import random
args = get_args()
from memo_chat.gpt_memochat import run_memochat
from memorybank.question_memorybank import run_memorybank, run_memorybank_response
#from memoryrecu.gpt_recu import run_memorybank, run_memorybank_response
min_len = 0
ave_len = 0
model_dict = {"baichuan": "baichuan_13B", 
              "chatglm":"chatglm2_6b",
              "vicuna":"vicuna-7b"}

prompts = json.load(open("data/msc_dialogue/prompts.json", "r"))

def build_model(args):
    model, tokenizer = None, None
    if "llama" in args.model_name:
        if args.model_name == 'llama2-13b-hf':
            path_name = "/data/zhuxiaowei/project/Mymodel/Llama-2-13b-hf"
        elif args.model_name == 'llama2-7b-hf':
            path_name = "/data/dl4nlp/pretrained_model/llama2-7b-hf"
        elif args.model_name == 'llama-7b-hf':
            path_name = "/data/zhuxiaowei/project/Mymodel/llama-7b-hf"
            args.max_seq_length = 2000
        elif args.model_name == 'llama2-13b-chat-hf':
            path_name = "/data/dl4nlp/pretrained_model/llama2-13b-chat-hf"
        elif args.model_name == 'llama2-7b-longlora-8k-ft':
            path_name = "/data/dl4nlp/pretrained_model/llama2-7b-longlora-8k-ft"
        
        tokenizer = AutoTokenizer.from_pretrained(path_name, padding_side="left")
        # print(tokenizer.padding_side)
        model = LlamaForCausalLM.from_pretrained(f'{path_name}', device_map='auto',  use_safetensors=False)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

    elif "longlora" in args.model_name:
        peft_model_path = "/data/dl4nlp/pretrained_model/llama2-7b-longlora-16k"
        base_model_path = "/data/dl4nlp/pretrained_model/llama2-7b-hf"
        config = AutoConfig.from_pretrained(base_model_path)
        model = LlamaForCausalLM.from_pretrained(f'{base_model_path}', device_map='auto',  use_safetensors=False)
        model.resize_token_embeddings(32001)
        trainable_params = os.path.join(peft_model_path, "trainable_params.bin")
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="right")
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        
    elif "baichuan" in args.model_name:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(f"/data/zhuxiaowei/project/Mymodel/{model_dict[args.model_name]}", trust_remote_code=True, revision="").cuda().half()
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"/data/zhuxiaowei/project/Mymodel/{model_dict[args.model_name]}", trust_remote_code=True, revision="")
        model.generation_config = GenerationConfig.from_pretrained(f"/data/zhuxiaowei/project/Mymodel/{model_dict[args.model_name]}")

    elif "chatglm" in args.model_name:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(f"/data/zhuxiaowei/project/Mymodel/{model_dict[args.model_name]}", trust_remote_code=True, revision="").cuda().half()
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"/data/zhuxiaowei/project/Mymodel/{model_dict[args.model_name]}", trust_remote_code=True, revision="")

    return model, tokenizer

def model_chat(args, inputs, model, tokenizer=None, logprobs=False):
    results = [] 
    for input in inputs:
        messages = [[{"role": "user", "content": input}]]
        if "llama" in args.model_name:
            response = model.chat_completion(messages,  # type: ignore
                max_gen_len=100,
                temperature=0,
                top_p=0.9,
                logprobs = logprobs
                )
            result = response[0]['generation']['content']
            logit = response[0]['logprobs']
        else:
            messages = [{"role": "user", "content": input}]
            result = model.chat(tokenizer, messages)
        results.append(result)
    return results, logit
            
def infer(args):
    model, tokenizer = build_model(args)
    _, test_loader = prepare_test_data(args) 
    pred_dict = {}
    for dialogs in tqdm(test_loader):
        inputs, labels, dial_ids = dialogs["input"], dialogs["label"], dialogs["dial_ids"]
        results, logits = model_chat(args, inputs, model, tokenizer, True)
        for dial_id, label, response in zip(dial_ids, labels, results):
            if dial_id in pred_dict.keys():
                continue
            try:
                response = response.split(":")[1].strip()
            except:
                response = response.strip()
            pred_dict[dial_id] = {"label":label, "prediction":response}
    
    wfile= os.path.join(args.saving_dir, f"{args.dataset}_{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dict, ensure_ascii=False, indent=4))  

    evaluate_file(args)

def sum_api(args):
    _, sum_test_dataset = load_dataset(args)
    pred_dict={}
    for dialog in tqdm(sum_test_dataset):
        histories, dial_id = dialog['history'], dialog['dial_id']
        pred_dict[dial_id] = []
        if dialog['pred_prev_summary'] == []:
            print(1)
            prev_memory = "EMPTY"
            for history in histories:
                prompt_str = prompts['gpt-3.5-turbo']['gen_memory1'].format_map({"prev_memory": prev_memory, "dialog": history})
                prev_memory = gpt_memory_results(prompt_str, args.model_name)
                pred_dict[dial_id].append(prev_memory)
        else:
            print(0)
            prev_memory = dialog['pred_prev_summary'][args.session_id-1]
            pred_dict[dial_id].append(prev_memory)
    
    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dict, ensure_ascii=False, indent=4))  
    #evaluate_file(args)


def rsum_dial(args):
    model, tokenizer = build_model(args)
    test_loader = prepare_rsum_data(args) 
    pred_dict={}   
    for dialogs in tqdm(test_loader):
        inputs, labels, dial_ids = dialogs["input"], dialogs["label"], dialogs["dial_ids"]
        results = model_chat(args, inputs, model, tokenizer)
        for dial_id, label, result in zip(dial_ids, labels, results):
            if dial_id in pred_dict.keys():
                continue
            if ":" not in result:
                response = result
            else:
                response = result.split(":")[1].strip()
            pred_dict[dial_id] = {"label":label, "prediction":response}
    
    wfile= os.path.join(args.saving_dir, f"{args.dataset}_{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dict, ensure_ascii=False, indent=4))  
    evaluate_file(args)

    
def evaluate_file(args, filename=False):
    dial_ids, labels, predictions = [],[],[]
    if filename:
        fileHandler = open(f"{filename}",  "r")
    else:
        fileHandler = open(f"{args.saving_dir}/{args.dataset}_{args.operation}_{args.mode}_sid{args.session_id}.json",  "r")
    pred_dicts = json.load(fileHandler)
    for keys,values in tqdm(pred_dicts.items()):
        dial_id, label, result = keys, values["label"], values["prediction"]
        if dial_id not in dial_ids:
            dial_ids.append(dial_id)
            labels.append(label)
            predictions.append(result)
    
    results = evaluate_corpus(predictions, labels)
    result = {k: round(v * 100, 4) for k, v in results.items()}
    print(result)


def chat_model(args):
    test_dataset = load_dataset(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = build_model(args)
    model.eval()
    nercollate = NerCollate(args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn = nercollate.collate_fn)
    all_preds = []
    all_trues = []
    pred_dicts = {}
    split_str = test_dataset.prompt.split("\n")[-1]
    #import pdb;pdb.set_trace()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, ncols=100)):
            for k,v in batch.items():
                batch[k] = v.to(device)
            #import pdb;pdb.set_trace()
            output = model.generate(**batch, max_new_tokens=100, temperature=0, top_p=0.9)
            decoded_preds = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
            for pred in decoded_preds:
                out = pred.split(split_str)[-1].replace("\n","").strip()
                if "User:" in out:
                    out = out.split("User:")[0]
                all_preds.append(out)
            #import pdb;pdb.set_trace()

    for (data, pred) in zip(test_dataset, all_preds):
        all_trues.append(data['response'])
        pred_dicts[data["dial_id"]] = {'prediction': pred, 'label': data['label']}
    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    results = evaluate_corpus(all_preds, all_trues)
    result = {k: round(v * 100, 4) for k, v in results.items()}
    print(result)
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(result)+"\n")

def compute_pll(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nlls = []
    model, tokenizer = build_model(args)
    model.eval()
    test_dataset = load_dataset(args)
    nercollate = NerCollate(args, tokenizer,"train")
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn = nercollate.collate_fn)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, ncols=100)):
            for k,v in batch.items():
                batch[k] = v.to(device)
            output = model(**batch)
            neg_log_likelihood = output.loss
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    ave_length = nercollate.total_tokens / len(test_dataset) * 1.0
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write('PPL:{:.2f} '.format(ppl))
        f.write('#Average Tokens: {:.2f} #Max Tokens: {:.2f}\n'.format(ave_length, nercollate.max_tokens))

def get_response_with_memory(args, test_dial):
    if args.operation == 'infer':
        input_str = prompts['gpt-3.5-turbo']['gen_response_with_memory2'].format_map(
            {"prev_memory": test_dial['pred_prev_summary'], "dialog": test_dial['window']})
    elif args.operation == 'ict':
        input_str = prompts['gpt-3.5-turbo']['gen_response_with_memory_example'].format_map(
            {"prev_memory": test_dial['pred_prev_summary'], "dialog": test_dial['window'], 'example': prompts['gpt-3.5-turbo']["example3"]})
    #import pdb; pdb.set_trace()
    response = gpt_response_results(args, input_str, args.model_name)
    return response


def get_response(args, test_dial):
    if args.mode == 'rag':
        input_str = prompts['gpt-3.5-turbo']['gen_response_with_rag'].format_map({"dialog": test_dial['window'], 'retrieved': test_dial['rag']})
    elif args.mode == 'rag_mem':
        input_str = prompts['gpt-3.5-turbo']['gen_response_with_rag_memory'].format_map({"dialog": test_dial['window'], 'retrieved': test_dial['rag'], 'prev_memory': test_dial['pred_prev_summary']})
    else:
        input_str = prompts['gpt-3.5-turbo']['gen_response'].format_map({"dialog": test_dial[args.mode]})
    #import pdb; pdb.set_trace()
    response = gpt_response_results(args, input_str, args.model_name)
    return response

def chat_api(args):
    test_data, test_dataset = load_dataset(args)

    all_preds = []
    all_trues = []
    pred_dicts = {}
    summary_list = []
    for idx, test_dial in enumerate(tqdm(test_dataset, ncols=100)):
        if args.mode == 'rsum':
            pred = get_response_with_memory(args, test_dial)
        else:
            pred = get_response(args, test_dial)
            
        label = test_dial['response']
        #import pdb; pdb.set_trace()
        if args.mode == 'rsum':
            pred_dicts[test_dial['dial_id']] = {'prediction': pred, 'label': test_dial['response'], 'summary_list': test_dial['pred_prev_summary'], 'summary_label': test_dial['gt_prev_summary_string']}
        else:
            pred_dicts[test_dial['dial_id']] = {'prediction': pred, 'label': test_dial['response']}
        all_preds.append(pred)
        all_trues.append(label)
    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    results = evaluate_corpus(all_preds, all_trues)
    result = {k: round(v * 100, 4) for k, v in results.items()}
    print(result)
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(result)+"\n")
    

def test(args, test_dataset):
    with open (f"save_msc/gpt-3.5-old/full_sid{args.session_id}_refs.txt","r") as fr:
        labels = fr.readlines()

    with open (f"save_msc/gpt-3.5-old/full_sid{args.session_id}_hyps.txt","r") as fr:
        predictions = fr.readlines()

    pred_dicts = {}
    for idx, dial in enumerate(test_dataset):
        dial_id = dial['dial_id']
        pred_dicts[dial_id] = {'prediction': predictions[idx], 'label': dial['response']}
    
    wfile= os.path.join("save_msc/gpt-3.5-old", f"infer_full_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4)) 

def evaluate_summary(args, test_dataset):
    labels, predictions = [],[]
    fileHandler = open(f"save_msc/memorybank/infer_full_sid4.json",  "r")
    pred_dicts = json.load(fileHandler)
    idx = 0
    for keys, values in tqdm(pred_dicts.items()):
        dial_id, persona, memory = keys, values["persona"], values['memory']
        result = persona + " " + memory 
        label = test_dataset[idx]['gt_prev_summary_string'].replace("I","").replace("am","")
        #import pdb; pdb.set_trace()
        labels.append(label)
        predictions.append(result)
        idx = idx + 1
    
    eval_matrics = evaluate_memory(predictions, labels)
    report_matrics = {k: round(v * 100, 4) for k, v in eval_matrics.items()}
    print(report_matrics)

if __name__ == "__main__":
    args = get_args()
    args.saving_dir = os.path.join("save", args.dataset, args.model_name)
    if not os.path.exists(args.saving_dir): 
        os.makedirs(args.saving_dir)

    args.logger_file = os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}_log.txt")
    test_data, test_dataset = load_dataset(args)
    if args.operation == 'judge':
        run_llm_judge(args, test_dataset)
        #load_eval_file(args)
    elif args.operation == 'win':
        run_llm_win(args, test_dataset)
    elif args.operation in ['infer','ict', 'rsum']:
        if 'gpt' in args.model_name:
            if args.mode == 'sum':
                sum_api(args)
            else:
                chat_api(args)
        elif 'memochat' in args.model_name:
            run_memochat(args, test_dataset)
        elif 'memorybank' in args.model_name:
            if 'question' in args.model_name:
                from memorybank.question_memorybank import run_memorybank
            elif 'retrieval' in args.model_name:
                from memorybank.retrieval_memorybank import run_memorybank
            else:
                from memorybank.gpt_memorybank import run_memorybank
            run_memorybank(args, test_dataset)
        else:
            chat_model(args)
    elif args.operation == 'eval':
        evaluate_summary(args, test_data)
        #compute_pll(args)
    '''if args.operation == "infer": 
        infer(args)
    elif "summary" in args.operation:
        summarize_dial(args)
    elif args.operation == "evaluate":
        if args.model_name == "longlora":
            evaluate_longlora(args)
        else:
            evaluate_file(args, True)
    elif "dialog" in args.operation:
        rsum_dial(args)'''










