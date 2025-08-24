import argparse
import logging

def get_args():
    parser = argparse.ArgumentParser()
    #dataset param
    parser.add_argument("--dataset", type=str, default="msc", help="msc or carecall")
    parser.add_argument("--session_id", type=int, default=5, help="1,2,3,4,5")
    parser.add_argument("--mode", type=str, default="full", help="full, window, rsum, rag, sum")
    parser.add_argument("--nopersona_subsampling_weight", type=float, default=0)
    parser.add_argument("--max_seq_length", type=int, default=4000)

    parser.add_argument("--summary_size", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=2000)
    parser.add_argument("--target_size", type=int, default=200)
    parser.add_argument("--resp_temp", type=float, default=0)
    parser.add_argument("--summ_temp", type=float, default=0)   
    parser.add_argument("--eval_file", type=str, default="msc_dialog_window_sid5.json", help="the location of prediction file")
    parser.add_argument("--summary_file", type=str, default="msc_sum_sid5", help="the location of summary file")

    parser.add_argument("--operation", type=str, default="judge", help="infer, finetune, evaluate, summary, dialog, dialog_ict, summary_ict")
    parser.add_argument("--do_rag", action='store_true')
    parser.add_argument("--do_ict", action='store_true')


    parser.add_argument("--retrieval", type=str, default="b25", help="dpr or b25")

    parser.add_argument("--saving_dir", type=str, default="save", help="the location of saving results")
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--random_seed", type=int, default=42, help="the random seed")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--dev_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--test_num", type=int, default=300, help="the number of test dialogue")
    parser.add_argument("--n_shot", type=int, default=1, help="the number of N-shot")
    parser.add_argument("--summary_type", type=str, default="pred", help="pred or gt")
    #
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--summary_model_name", type=str, default="gpt-3.5-turbo-0301", help="Choose model for inference")
  
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13", help="llama2-13b-chat or llama2-7b-chat for inference")
    parser.add_argument("--trainer", type=str, default="summarizer", help="summarizer or dialog")
    parser.add_argument("--summary_num_turns", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=3)

    #load
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--topk", type=int, default=5)


    args = parser.parse_args()
    return args

def get_logger(file_log, fh_mode="w"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=file_log, encoding='utf-8', mode=fh_mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

