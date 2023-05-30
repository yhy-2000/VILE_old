# coding=utf-8
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
sys.path.append("./")
import argparse
import subprocess
import logging
import os
import numpy as np
import torch
import faiss
from transformers import RobertaConfig,BertConfig
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import copy
import json
from dataset import (
    TextTokenIdsCache, load_rel, SubsetSeqDataset, SequenceDataset,
    single_get_collate_function
)
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)
logger = logging.Logger(__name__)
from transformers import RobertaModel,BertModel
from model import EncoderDot

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value




def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        print(f"using {args.n_gpu} gpus to inference.......")
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)


def query_inference(model, args, embedding_size):
    # if os.path.exists(args.query_memmap_path):
    #     print(f"{args.query_memmap_path} exists, skip inference")
    #     return

    query_collator = single_get_collate_function(args.max_query_length)
    query_preprocess_dir=os.path.join(args.preprocess_dir,"query")

    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=query_preprocess_dir, prefix=f"{args.mode}-query",args=args),
        max_seq_length=args.max_query_length,
        config=args
    )
    query_memmap = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.queryids_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(query_dataset), ))
    try:
        prediction(model, query_collator, args,
                query_dataset, query_memmap, queryids_memmap, is_query=True)
    except Exception as e:
        subprocess.check_call(["rm", args.query_memmap_path])
        subprocess.check_call(["rm", args.queryids_memmap_path])
        raise


def doc_inference(model, args, embedding_size):
    # if os.path.exists(args.doc_memmap_path):
    #     print(f"{args.doc_memmap_path} exists, skip inference")
    #     return
    
    doc_collator = single_get_collate_function(args.max_root_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages",args=args)
    subset=list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_root_length,
        leaf=True,
        config=args
    )
    doc_memmap = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(doc_dataset), ))
    try:
        prediction(model, doc_collator, args,
            doc_dataset, doc_memmap, docid_memmap, is_query=False
        )
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.docid_memmap_path])
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_s_type", type=str, default='leaf')
    parser.add_argument("--sample_type", type=str, default='inbatch')
    parser.add_argument("--init_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=str, default=None)
    parser.add_argument("--pretrain_model", type=str, default="retromae")
    parser.add_argument("--from_small_to_big", type=bool, default=False)
    input_args = parser.parse_args()

    if input_args.init_path==None:
        input_args.init_path="./drhard/data/doc_subset/bert/"+input_args.sample_type+"_train/"+input_args.model_s_type+"/"+input_args.pretrain_model+"/"
        if input_args.learning_rate!=None:
            input_args.init_path+=input_args.learning_rate+"/"
    if "checkpoint" in input_args.init_path:
        args_path="/".join(input_args.init_path.split("/")[:-1])+"/training_args.json"
    else:
        args_path=input_args.init_path+"/training_args.json"


    args=DotDict(json.load(open(args_path,"r")))
    args.init_path=input_args.init_path
    print("1111111",args.init_path)

    args.mode="dev"
    args.eval_batch_size=32
    args.faiss_gpus=[0]
    args.topk=100
    # assert args.model_s_type==input_args.model_s_type

    if args.init_path[-1]==['/']:args.init_path=args.init_path[:-1]
    args.output_dir = "./drhard/data/evaluate/doc_subset/"+args.init_path.split("/")[-4]+"/"+args.init_path.split("/")[-3]+"/"+args.init_path.split("/")[-2]+"/"+args.init_path.split("/")[-1]

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.preprocess_dir = "./drhard/data/doc_subset/preprocess_multimodal_new/"

    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    config = BertConfig.from_pretrained(args.init_path, gradient_checkpointing=False)

    bert_model = BertModel.from_pretrained(args.init_path,config=config, add_pooling_layer=False)

    model = EncoderDot(bert_model,args)
    checkpoint = torch.load(os.path.join(args.init_path, "pytorch_model.bin"))
    model.load_state_dict(checkpoint)

    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)

    query_inference(model, args, output_embedding_size)
    doc_inference(model, args, output_embedding_size)
    
    model = None
    torch.cuda.empty_cache()

    doc_embeddings = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryids_memmap_path, 
        dtype=np.int32, mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)

    with open(args.output_rank_file, 'w') as outputfile:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")

    os.system("python ./drhard/star_hierarchical/msmarco_eval_my.py --path_to_reference ./drhard/data/doc_subset/preprocess_multimodal_new/query/dev-qrel.tsv --path_to_candidate %s/dev.rank.tsv"%args.output_dir)

if __name__ == "__main__":
    main()



