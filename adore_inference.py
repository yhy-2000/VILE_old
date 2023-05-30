import sys
sys.path += ["./"]
import os
import torch
import faiss
import logging
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaConfig,BertConfig,BertModel
from model import EncoderDot
from dataset import TextTokenIdsCache, SequenceDataset, get_collate_function,single_get_collate_function
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def evaluate(args, model):
    """ Train the model """    

    query_preprocess_dir=os.path.join(args.preprocess_dir,"query")
    dev_dataset = SequenceDataset(
            TextTokenIdsCache(query_preprocess_dir, f"{args.mode}-query",args), 
            args.max_seq_length,
            args)
    # collate_fn = get_collate_function(args.max_seq_length)
    collate_fn = single_get_collate_function(args.max_seq_length)
    batch_size = args.pergpu_eval_batch_size
    if args.n_gpu > 1:
        batch_size *= args.n_gpu
    dev_dataloader = DataLoader(dev_dataset, 
        batch_size= batch_size, collate_fn=collate_fn)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    qembedding_memmap = np.memmap(args.qmemmap_path, dtype="float32",
        shape=(len(dev_dataset), 768), mode="w+")
    with torch.no_grad():
        for step, (batch, qoffsets) in enumerate(tqdm(dev_dataloader)):
            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.eval()            
            embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                is_query=True)
            embeddings = embeddings.detach().cpu().numpy()
            qembedding_memmap[qoffsets] = embeddings
    return qembedding_memmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./drhard/data/doc_subset/bert/adore/epoch-6")
    parser.add_argument("--output_dir", type=str, default="./drhard/data/evaluate/doc_subset/adore")
    parser.add_argument("--preprocess_dir", type=str, default="./drhard/data/doc_subset/preprocess_multimodal_new")
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], default="dev")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--dmemmap_path", type=str, default="./drhard/data/evaluate/doc_subset/star_train/leaf/bert/1e-06/passages.memmap")
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--pergpu_eval_batch_size", type=int, default=32)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--faiss_gpus", type=int, default=None, nargs="+")
    args = parser.parse_args()
    logger.info(args)

    assert os.path.exists(args.dmemmap_path)
    os.makedirs(args.output_dir, exist_ok=True)
    # Setup CUDA, GPU 
    args.use_gpu = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", args.device, args.n_gpu)
    config = BertConfig.from_pretrained(args.model_dir, gradient_checkpointing=False)
    bert_model = BertModel.from_pretrained(args.model_dir,config=config, add_pooling_layer=False)

    model = EncoderDot(bert_model,args)
    checkpoint = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"))
    model.load_state_dict(checkpoint,strict=False)
    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)


    logger.info("Training/evaluation parameters %s", args)
    # Evaluation
    args.qmemmap_path = f"{args.output_dir}/{args.mode}.qembed.memmap"
    evaluate(args, model)
    
    doc_embeddings = np.memmap(args.dmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)

    query_embeddings = np.memmap(args.qmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)
    model = None
    torch.cuda.empty_cache()

    index = construct_flatindex_from_embeddings(doc_embeddings, None)
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)
    output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    with open(output_rank_file, 'w') as outputfile:
        for qid, neighbors in enumerate(nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")