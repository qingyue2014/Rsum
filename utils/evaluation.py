import collections
from nltk import ngrams
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re
from bert_score import BERTScorer
from rouge import Rouge
import tiktoken

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
bert_score_path = "/data/dl4nlp/pretrained_model/roberta-large"

def compute_f1(preds, labels):
    if preds == []:
        return 0
    sum_f1 = 0
    sum_precsion = 0
    sum_recall = 0
    for a_pred, a_gold in zip(preds, labels):
        gold_toks = word_tokenize(normalize_answer(a_gold))
        pred_toks = word_tokenize(normalize_answer(a_pred))
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            sum_f1 += int(gold_toks == pred_toks)
            continue
        if num_same == 0:
            sum_f1 += 0
            continue
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        sum_precsion += precision
        sum_recall += recall
        sum_f1 += (2 * precision * recall) / (precision + recall)
    f1_score = round(sum_f1/len(preds), 5)
    precision_score = round(sum_precsion/len(preds), 5)
    recall_score = round(sum_recall/len(preds), 5)
    return f1_score, precision_score, recall_score

def compute_f1_sentence(pred, label):
    gold_toks = normalize_answer(label).split()
    pred_toks = normalize_answer(pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def compute_acc(preds, labels):
    if preds == []:
        return 0
    sum_f1 = 0
    for a_pred, a_gold in zip(preds, labels):
        gold_toks = a_gold.strip().split(" ")
        pred_toks = a_pred.strip().split(" ")
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = len(common)
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            sum_f1 += int(gold_toks == pred_toks)
            continue
        if num_same == 0:
            sum_f1 += 0
            continue
        recall = 1.0 * num_same
        sum_f1 += recall
    acc = round(sum_f1/len(labels), 5)
    return acc


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(nltk.ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def calculate_distinct_n(corpus, n):
    # 分词和去停用词
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
    filtered_sentences = [[word for word in sentence if word.isalnum()] for sentence in tokenized_sentences]

    # 创建n-gram
    n_grams = [list(ngrams(sentence, n)) for sentence in filtered_sentences]

    # 计算独特n-gram数
    unique_n_grams = set([gram for sentence_n_grams in n_grams for gram in sentence_n_grams])

    # 计算distinct-n
    distinct_n = len(unique_n_grams) / sum(len(sentence_n_grams) for sentence_n_grams in n_grams) * 100


    return distinct_n

def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate.lower()) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score

def calc_distinct(candidates, print_score: bool = False):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def compute_bleu(predictions, labels):
    inf_tok = [nltk.tokenize.word_tokenize(i) for i in labels]
    tra_tok = [nltk.tokenize.word_tokenize(t) for t in predictions]
    chencherry =  SmoothingFunction()
    weights = [(0.5, 0.5), 
               (0.333, 0.333, 0.334),
               (0.25, 0.25, 0.25, 0.25),
               (0.2, 0.2, 0.2, 0.2, 0.2)]
    return corpus_bleu(inf_tok, tra_tok, weights=weights, smoothing_function=chencherry.method7)

def evaluate_corpus(predictions, references):
    f1_score, _, _ = compute_f1(predictions, references)
    dist_score = calc_distinct(predictions)
    bleu_score = compute_bleu(predictions, references)
    bleu_ave = (bleu_score[0] + bleu_score[1] + bleu_score[2]) / 3
    rouge = Rouge()
    return {"F1":f1_score,"Dist1":dist_score[0], "Dist2":dist_score[1], "BlEU1": bleu_score[0], 
            "BLEU2":bleu_score[1], "BLEU3":bleu_score[2], 'bleu_ave': bleu_ave}


def evaluate_length(predictions):
    length = 0
    for pred in predictions:
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
        length += len(encoding.encode(pred))  # token num
    ave = length / len(predictions) * 1.00
    print(ave)

def evaluate_memory(predictions, references):
    f1_score, precision, recall = compute_f1(predictions, references)
    dist_score = calc_distinct(predictions)
    bleu_score = compute_bleu(predictions, references)
    bleu_ave = (bleu_score[0] + bleu_score[1] + bleu_score[2]) / 3
    return {"F1":f1_score, "precision": precision, "recall": recall, "Dist1":dist_score[0], "Dist2":dist_score[1], "BlEU1": bleu_score[0], 
            "BLEU2":bleu_score[1], "BLEU3":bleu_score[2], 'bleu_ave': bleu_ave}
