# # coding=utf-8
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import wandb

# wandb.init(
# project='yhy1',
# entity='229-163',
# )

import os
os.environ['WANDB_SILENT']="true"
os.environ['WANDB_DISABLED']="true"

import json
import imp
import sys
sys.path.append("./")
import logging
import os
from dataclasses import dataclass, field
import transformers
from transformers import (
    BertConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import torch
from transformers.integrations import TensorBoardCallback
from dataset import TextTokenIdsCache, load_rel
from dataset import (
    TrainInbatchDataset, 
    TrainInbatchWithHardDataset,
    TrainInbatchWithRandDataset,
    triple_get_collate_function,
    dual_get_collate_function
)
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    Trainer, 
    TrainerCallback, 
    TrainingArguments, 
    TrainerState, 
    TrainerControl,
    RobertaConfig,
    BertConfig,
    BertTokenizer,
    BertModel
    )
from star_tokenizer import RobertaTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup,RobertaModel
from lamb import Lamb

import argparse
import copy

logger = logging.Logger(__name__)

# import model_independent_encoder,model_new_position_embedding,model_new_token_type_id,model_new_position_embedding,model_pure_text
from model import EncoderDot_InBatch

class MyTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


class DRTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer_str == "adamw":
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer_str == "lamb":
                self.optimizer = Lamb(
                    optimizer_grouped_parameters, 
                    lr=self.args.learning_rate, 
                    eps=self.args.adam_epsilon
                )
            else:
                raise NotImplementedError("Optimizer must be adamw or lamb")
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
    

class MyTensorBoardCallback(TensorBoardCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        pass


def is_main_process(local_rank):
    return local_rank in [-1, 0]

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./drhard/data/tmp") # where to output
    logging_dir: str = field(default=None)
    padding: bool = field(default=False)
    optimizer_str: str = field(default="lamb") # or lamb
    overwrite_output_dir: bool = field(default=True)    
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},)

    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=40, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=5000, metadata={"help": "Save checkpoint every X updates steps."})
    
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    query_type: int = field(default=1) # 0 for original queries, 1 for subset queries we construct
    with_image: bool = field(default=True) # 0 for original queries, 1 for subset queries we construct
    # train_from_ckpt: str = field(default="./drhard/data/tmp/checkpoint-1116")
    train_from_ckpt: str = field(default=None)
    init_path: str = field(default=None) # please use bm25 warmup model or roberta-base
    # init_path: str = field(default="./drhard/data/doc_subset/bert/inbatch_train/baseline/") # train from a better

    freeze_text: bool = field(default=False) 
    gradient_checkpointing: bool = field(default=True)
    max_query_length: int = field(default=32) # 24
    max_root_length: int = field(default=256) #  512 for doc and 120 for passage
    max_block_length: int = field(default=128) #  512 for doc and 120 for passage
    hardneg_path: str = field(default="./drhard/data/doc_subset/hard_neg/bm25_retrieve/hard.json") # use prepare_hardneg.py to generate
    block_num: int = field(default=8)
    model_type: str = field(default="leaf") # choose from ["tree","leaf","pure_text"]
    sample_type: str = field(default="inbatch")# choose from ["inbatch","inbatch_bm25","ance","star"]
    loss: str = field(default="cross_entropy")# choose from ["cross_entropy","cross_entropy_tevatron"]
    model_s_type: str = field(default="leaf") # choose from ["pure_text","leaf","new_token_type_id","new_position_embedding","independent_encoder"]
    pretrain_model: str = field(default="bert") # choose from ["contriever","condenser","retromae"]
    use_page_encoder: bool = field(default=True) 
  

def print_weight(model,path):
    fw=open(path,"w")
    for param in model.parameters():
        print(param.data,file=fw)

def main():
    parser = HfArgumentParser(MyTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.dataloader_num_workers=8

    args.logging_dir=os.path.join(args.output_dir,"log")
    

    args.preprocess_dir="./drhard/data/doc_subset/preprocess_multimodal_new"

    if args.model_s_type=="pure_text":
        args.model_type="pure_text"
    else:
        args.model_type="leaf"
        
    if args.model_s_type=="no_hierarchical":
        args.block_num=1     

    # 没有指定默认生成
    if "tmp" in args.output_dir:
        args.output_dir=os.path.join("./drhard/data/doc_subset/bert",args.sample_type+"_train",args.model_s_type,args.pretrain_model,str(args.learning_rate))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.output_dir+"/training_args.json","w") as fw:
        fw.write(args.to_json_string())
    print(args.to_json_string())
        
    # record model scripts and load write model
    os.system("cp ./drhard/star_hierarchical/model.py "+args.output_dir)
    

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    if args.pretrain_model=="contriever":
        args.pretrain_path="facebook/contriever"
    elif args.pretrain_model=="contriever_msmarco":
        args.pretrain_path="facebook/contriever-msmarco"
    elif args.pretrain_model=="retromae":
        args.pretrain_path="./drhard/data/pretrained_model/retromae"
    elif args.pretrain_model=="retromae_msmarco":
        args.pretrain_path="./drhard/data/pretrained_model/retromae_msmarco"
    elif args.pretrain_model=="retromae_msmarco_finetune":
        args.pretrain_path="./drhard/data/pretrained_model/retromae_msmarco_finetune"
    elif args.pretrain_model=="condenser":
        args.pretrain_path="Luyu/condenser"
    elif args.pretrain_model=="condenser_msmarco":
        args.pretrain_path="Luyu/co-condenser-marco"
    elif args.pretrain_model=="bert":
        args.pretrain_path="./drhard/data/pretrained_model/bert"
    else:
        raise KeyError

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    config = BertConfig.from_pretrained(
        args.pretrain_path,
        finetuning_task="msmarco",
        gradient_checkpointing=args.gradient_checkpointing,
        return_dict=False
    )
  
    query_preprocess_dir=os.path.join(args.preprocess_dir,"query")
    args.label_path = os.path.join(args.preprocess_dir,"query", "train-qrel.tsv")
    rel_dict = load_rel(args.label_path)

    gpu_num = torch.cuda.device_count()
    data_collator = triple_get_collate_function(
        args.max_query_length, args.max_root_length,
        rel_dict=rel_dict, padding=args.padding,gpu_num=gpu_num)

    bert_model = BertModel.from_pretrained(args.pretrain_path, config=config, add_pooling_layer=False)
    model = EncoderDot_InBatch(bert_model,args)
    
    # train from warmup, the qry_encoder and doc_encoder are different, and linear will also be initialized
    if args.init_path is not None:        
        model.load_state_dict(torch.load(os.path.join(args.init_path,"pytorch_model.bin")))


    train_dataset = TrainInbatchWithHardDataset(
        rel_file=args.label_path,
        rank_file=args.hardneg_path,
        queryids_cache=TextTokenIdsCache(data_dir=query_preprocess_dir, prefix="train-query",args=args),
        docids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages",args=args),
        hard_num=1,
        config=args
    )

    # Initialize our Trainer
    trainer = DRTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.remove_callback(TensorBoardCallback)
    trainer.add_callback(MyTensorBoardCallback(
        tb_writer=SummaryWriter(os.path.join(args.output_dir, "log"))))
    trainer.add_callback(MyTrainerCallback())

    # trainer.train()
    if args.train_from_ckpt:
        ckpt_dir=args.train_from_ckpt
    else:
        ckpt_dir=None
    trainer.train(ckpt_dir)
    trainer.save_model()  # Saves the tokenizer too for easy upload


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
