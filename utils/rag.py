import torch
import torch.nn.functional as F
from transformers import(
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)

import time
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
dpr_path = "/home/data/dpr-ctx_encoder-multiset-base"
q_encoder = DPRContextEncoder.from_pretrained(dpr_path).to(device=device)
q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(dpr_path)
d_encoder = DPRQuestionEncoder.from_pretrained(dpr_path).to(device=device)
d_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(dpr_path)


def embed(sentence, type='q'):
    if type == 'q':
        q_input_ids = q_tokenizer(
            sentence,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )["input_ids"]
        return q_encoder(q_input_ids.to(device=device), return_dict=True).pooler_output.detach().cpu()
    else:
        d_input_ids = d_tokenizer(
            "",     # title
            sentence,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]
        return d_encoder(d_input_ids.to(device=device), return_dict=True).pooler_output.detach().cpu()


def rag(data, topk=3):
    s_time = time.time()
    for i, data_dict in tqdm(enumerate(data)):
        docs = data_dict['full']
        assert (len(docs)-1) % 2 == 0
        documents_emb = []
        documents = []
        tmp_content = ''
        for doc in docs[:-1]:
            if doc['role'] == 'user':
                tmp_content = doc['content']
            else:
                tmp_content += doc['content']
                documents.append(tmp_content)
                documents_emb.append(embed(tmp_content, type='d'))
                tmp_content = ''
        query = data_dict['window'][-1]['content']
        query_emb = embed(query, type='q')  # 1,768
        documents_emb = torch.stack(documents_emb).squeeze()  # N,768

        score = F.cosine_similarity(query_emb, documents_emb, dim=1)
        index_sorted = torch.topk(score, k=topk, largest=True)[1].tolist()

        data[i]['rag_result_context'] = [documents[idx] for idx in index_sorted]
        data[i]['rag_result_index'] = index_sorted

    e_time = time.time()
    print("Retrieval finish in " + str(e_time-s_time))
    return data
