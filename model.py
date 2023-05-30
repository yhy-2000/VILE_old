import enum
import sys
import copy
import torch.nn.init as init

from torch._C import dtype
sys.path += ['./']
import torch
from torch import nn
import transformers
if int(transformers.__version__[0]) <=3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
else:
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaModel,RobertaTokenizer,BertModel
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pyserini.index.lucene
from transformers import BertTokenizer

BACKBONE_PRETRAINED_MODEL=BertPreTrainedModel
BACKBONE_MODEL=BertModel

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from BACKBONE_MODEL to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        if not isinstance(emb_all, tuple):
            emb_all=emb_all.to_tuple()
        assert isinstance(emb_all, tuple),type(emb_all)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask,image_features,tree_attention_mask):
        raise NotImplementedError("Please Implement this method")


class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask,image_features,tree_attention_mask):
        # TODO should raise NotImplementedError
        # temporarily do this  
        return None 

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask,image_features=None,text_type="query")
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask, image_features,tree_attention_mask=None):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask,image_features=image_features,tree_attention_mask=tree_attention_mask,text_type="doc")
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def forward(self,is_query, input_ids, attention_mask, image_features=None,tree_attention_mask=None):
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask,image_features,tree_attention_mask)


class EncoderDot(BaseModelDot,BACKBONE_PRETRAINED_MODEL):
    def __init__(self, bert_model, args=None):
        BaseModelDot.__init__(self, None)
        BACKBONE_PRETRAINED_MODEL.__init__(self, bert_model.config)

        self.args = args
        config = bert_model.config

        self.query_encoder = copy.deepcopy(bert_model)
        self.doc_encoder = copy.deepcopy(bert_model)

        
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size

        # linear
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        
        # image encoder
        config.image_encoder_layer=3
        self.image_hidden_size=512
        # image_encoder_layers=[nn.Linear(self.image_hidden_size,config.hidden_size,dtype=torch.float32),nn.ReLU()]
        # image_encoder_layers.extend([nn.Linear(config.hidden_size,config.hidden_size,dtype=torch.float32),nn.ReLU()]*(config.image_encoder_layer-1))
        image_encoder_layers=[nn.Linear(self.image_hidden_size,config.hidden_size),nn.ReLU()]
        image_encoder_layers.extend([nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU()]*(config.image_encoder_layer-1))
        self.image_encoder = nn.Sequential(*image_encoder_layers)
        
        # initialize
        for module in self.image_encoder:
            if type(module)==nn.modules.linear.Linear:
                init.xavier_normal_(module.weight)
        init.xavier_normal_(self.embeddingHead.weight)

    def _text_encode(self, input_ids, attention_mask,image_features,tree_attention_mask=None,text_type="query"):

        # attention_mask[attention_mask>1]=1

        if text_type=="query":
            outputs1 = self.query_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)
            return outputs1
        else:

            if self.args.model_s_type=="pure_text": #只用父节点的input_ids
                page_output = self.doc_encoder(input_ids=input_ids[:,0,:],attention_mask=attention_mask[:,0,:])
                return page_output
            else:
                batch_size,block_num,seq_length = input_ids.shape
                image_tokens_num = 12
                device = input_ids.device
                # image_features: [bsz, block_num, 12, 512]
                if self.args.with_image:
                    tmp = torch.sum(image_features.reshape(image_features.shape[0]*image_features.shape[1],-1),dim=-1)
                    image_mask = tmp.eq(0).reshape(batch_size,block_num).to(device)
                    image_mask = 1-image_mask.long()
                else: 
                    image_mask = torch.zeros((batch_size,block_num),device=device)
                # print("aaaaaaaa",torch.sum(image_features))
                image_features = image_features.reshape(-1,self.image_hidden_size).to(torch.float32)
                image_features = F.normalize(image_features,dim=-1)
                image_embeddings = self.image_encoder(image_features)
                image_embeddings = image_embeddings.reshape(batch_size,block_num,image_tokens_num,768)
                            
                # truncate block_ids
                page_input_ids = input_ids[:,0,:]
                page_input_embeddings = self.doc_encoder.embeddings(page_input_ids)
                page_attn_mask =  attention_mask[:,0,:]
                page_image_embeddings = image_embeddings[:,0,:,:].reshape(-1,image_tokens_num,768)
                
                block_input_ids = input_ids[:,1:,:].reshape(batch_size*(block_num-1),-1)
                blocks_input_embeddings = self.doc_encoder.embeddings(block_input_ids).reshape(batch_size*(block_num-1),-1,768)
                blocks_attn_mask =  attention_mask[:,1:,:].reshape(batch_size*(block_num-1),-1)[:,:]  
                blocks_image_embeddings = image_embeddings[:,1:,:,:].reshape(-1,image_tokens_num,768)
                blocks_image_attn_mask = image_mask[:,1:].reshape(-1).unsqueeze(-1).repeat(1,image_tokens_num)

                # [CLS IMAGE*12 SEQ]
                block_input_embeddings = torch.cat((blocks_input_embeddings,blocks_image_embeddings),dim=1)
                block_attn_mask = torch.cat((blocks_attn_mask,blocks_image_attn_mask),dim=1)
                
                block_token_type_ids = torch.zeros(blocks_input_embeddings.shape[0],block_input_embeddings.shape[1],dtype=torch.int64,device=block_attn_mask.device)
                if self.args.pretrain_model=="bert":
                    block_token_type_ids[:,-12:]=1
                
                blocks_embedding= self.doc_encoder(inputs_embeds=block_input_embeddings,attention_mask=block_attn_mask,token_type_ids=block_token_type_ids)[0]

                # 采用pool的方式集成各个block
                blocks_embedding = blocks_embedding.reshape(batch_size,-1,blocks_embedding.shape[-2],768)
                
                # block_mask
                block_mask = torch.tensor([0 if attention_mask[bid][block_id][1]==0 else 1 for bid in range(batch_size) for block_id in range(blocks_embedding.shape[1])],device=device,dtype=torch.bool)
                block_mask = block_mask.reshape(batch_size,blocks_embedding.shape[1])
                blocks_embedding = blocks_embedding[:,:,0,:].reshape(batch_size,-1,768)
                
                # page embedding
                page_input_embeddings = torch.cat((page_input_embeddings,blocks_embedding,page_image_embeddings),dim=1)
                page_image_attn_mask = image_mask[:,0].reshape(-1).unsqueeze(-1).repeat(1,image_tokens_num)
                block_attn_mask = block_mask
                page_attn_mask = torch.cat((page_attn_mask,block_attn_mask,page_image_attn_mask),dim=1)

                page_token_type_ids = torch.zeros(page_attn_mask.shape[0],page_attn_mask.shape[1],dtype=torch.int64,device=block_attn_mask.device)
                if self.args.pretrain_model=="bert":
                    page_token_type_ids[:,-12:]=1                
                page_output = self.doc_encoder(inputs_embeds=page_input_embeddings,attention_mask=page_attn_mask,token_type_ids=page_token_type_ids)
               
                outputs1 = page_output
                return outputs1


class EncoderDot_InBatch(EncoderDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask=None,
            other_doc_ids=None, other_doc_attention_mask=None, other_doc_image_features=None, other_doc_tree_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        if self.args.sample_type=="inbatch_bm25"or self.args.sample_type=="ance":
            loss = inbatch_train_my(self.query_emb, self.body_emb,
                input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask,
                other_doc_ids, other_doc_attention_mask,other_doc_image_features,other_doc_tree_attention_mask,
                rel_pair_mask, hard_pair_mask,self.args)
        elif self.args.sample_type=="inbatch": # only consider inbatch neg
            loss = inbatch_train_my(self.query_emb, self.body_emb,
                input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask,
                None,None,None,None,
                rel_pair_mask, hard_pair_mask,self.args)  
        elif self.args.sample_type=="star":
            loss = inbatch_train(self.query_emb, self.body_emb,
                input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask,
                other_doc_ids, other_doc_attention_mask,other_doc_image_features,other_doc_tree_attention_mask,
                rel_pair_mask, hard_pair_mask,self.args)
              
        return loss


class EncoderDot_Rand(EncoderDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,
            other_doc_ids=None, other_doc_attention_mask=None,other_doc_image_features=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,
            other_doc_ids, other_doc_attention_mask, other_doc_image_features,
            hard_pair_mask)








def inbatch_neg_infoNCE(query_embs,doc_embs):
    # listwise infoNCE
    batch_size=query_embs.shape[0]
    logits=torch.matmul(query_embs,doc_embs.T)

    # let dim0 all positives
    pos_scores=torch.diagonal(logits).unsqueeze(-1)
    digonal_index=torch.eye(batch_size)
    neg_scores=logits[digonal_index!=1].reshape(batch_size,-1)
    logits=torch.cat((pos_scores,neg_scores),dim=-1)

    temperature=0.1
    logits/=temperature

    loss = -1*F.log_softmax(logits,dim=-1)[0]
    loss = loss.sum()/batch_size
    return loss





# 加上hard neg试试
def cross_entropy_loss(query_embs,doc_embs,other_doc_embs):
    with autocast(enabled=False):
        batch_size= query_embs.shape[0]
        score = torch.matmul(query_embs, doc_embs.T)
        label = torch.arange(batch_size, device=query_embs.device)
        if other_doc_embs!=None:
            hard_neg_score=torch.matmul(query_embs,other_doc_embs.T)
            score = torch.cat((score,hard_neg_score),dim=-1)
        loss = F.cross_entropy(score, label)
        # loss = F.cross_entropy(score, label,reduction="sum")

    return loss



def inbatch_train_my(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask=None,
            other_doc_ids=None, other_doc_attention_mask=None, other_doc_image_features=None,other_doc_tree_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None,args=None):

    batch_size,block_num,seq_length=input_doc_ids.shape
    image_token_num=12
    doc_image_features=doc_image_features.reshape(batch_size,block_num,image_token_num,-1)
    doc_tree_attention_mask=doc_tree_attention_mask.reshape(batch_size,block_num,block_num) if doc_tree_attention_mask is not None else None



    query_embs = query_encode_func(input_query_ids, query_attention_mask) # (bs,768)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask,doc_image_features,doc_tree_attention_mask) # (bs,768)

    other_doc_embs=None
    if other_doc_ids!=None:
        other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
        other_doc_ids = other_doc_ids.reshape(other_doc_num,block_num, -1)
        other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num,block_num, -1)
        
        other_doc_image_features = other_doc_image_features.reshape(other_doc_num,block_num,image_token_num,-1)
        other_doc_tree_attention_mask=other_doc_tree_attention_mask.reshape(batch_size,block_num,block_num)  if other_doc_tree_attention_mask is not None else None
        other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask,other_doc_image_features,other_doc_tree_attention_mask)
    
    if args.loss=="cross_entropy":
        loss=cross_entropy_loss(query_embs,doc_embs,other_doc_embs)

    return (loss,)



def inbatch_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,doc_tree_attention_mask=None,
            other_doc_ids=None, other_doc_attention_mask=None, other_doc_image_features=None,other_doc_tree_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None,args=None):

    batch_size,block_num=input_doc_ids.shape[0],input_doc_ids.shape[1]
    image_token_num=12
    doc_image_features=doc_image_features.reshape(batch_size,block_num,image_token_num,-1)
    doc_tree_attention_mask=doc_tree_attention_mask.reshape(batch_size,block_num,block_num) if doc_tree_attention_mask is not None else None
    
    # tokenizer=BertTokenizer.from_pretrained("./drhard/data/bert") 
    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask,doc_image_features,doc_tree_attention_mask)
    
    # print(query_embs.shape,doc_embs.shape)
    batch_size = query_embs.shape[0]
    rel_pair_mask = rel_pair_mask.squeeze()
    hard_pair_mask = hard_pair_mask.squeeze()
    if rel_pair_mask.shape[0]!=batch_size:
        rel_pair_mask=rel_pair_mask[:batch_size,:batch_size]
    if hard_pair_mask.shape[0]!=batch_size:
        hard_pair_mask=hard_pair_mask[:batch_size,:batch_size]
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T) #(bs,bs)
        single_positive_scores = torch.diagonal(batch_scores, 0) #(bs,1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)#(bs*bs)
        
        if rel_pair_mask is None:
            rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)                
        
        batch_scores = batch_scores.reshape(-1)
        logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                batch_scores.unsqueeze(1)], dim=1)
        lsm = F.log_softmax(logit_matrix, dim=1)  #(bs*bs,2)

        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
        first_loss, first_num = loss.sum(), rel_pair_mask.sum()


    if other_doc_ids is None:
        return (first_loss/first_num,)


    # other_doc_ids: batch size, per query doc, length

    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num,block_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num,block_num, -1)
    
    other_doc_image_features = other_doc_image_features.reshape(other_doc_num,block_num,image_token_num,-1)
    other_doc_tree_attention_mask=other_doc_tree_attention_mask.reshape(batch_size,block_num,block_num)  if other_doc_tree_attention_mask is not None else None
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask,other_doc_image_features,other_doc_tree_attention_mask)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)  
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    

    return ((first_loss+second_loss)/(first_num+second_num),)


def randneg_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, doc_image_features,
            other_doc_ids=None, other_doc_attention_mask=None,other_doc_image_features=None,
            hard_pair_mask=None):

    batch_size,block_num=input_doc_ids.shape[0],input_doc_ids.shape[1]
    image_token_num=12
    doc_image_features=doc_image_features.reshape(batch_size,block_num,image_token_num,-1)
    # doc_tree_attention_mask=doc_tree_attention_mask.reshape(batch_size,block_num,block_num) if doc_tree_attention_mask is not None else None

    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask,doc_image_features)
    
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        single_positive_scores = torch.diagonal(batch_scores, 0)
    # other_doc_ids: batch size, per query doc, length
    # other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    # other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    # other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    # other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask,other_doc_image_features)

    neg_per_qry = other_doc_ids.shape[1]
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num,block_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num,block_num, -1)
    
    # TODO: 目前不支持hard_neg>1的情况
    other_doc_image_features = other_doc_image_features.reshape(other_doc_num,block_num,image_token_num,-1)
    # other_doc_tree_attention_mask=other_doc_tree_attention_mask.reshape(batch_size,block_num,block_num)  if other_doc_tree_attention_mask is not None else None
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask,other_doc_image_features)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)  
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    return (second_loss/second_num,)
