"""
This is official eval script opensourced on MSMarco site (not written or owned by us)

This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
"""
I (Jingtao Zhan) modified this script for evaluating MSMARCO Doc dataset. --- 4/19/2021
"""
import sys
import statistics

from collections import Counter
import argparse
import os

MaxMRRRank_li = [10,100]
MaxRECALLRank_li = [1,10,100]

EVAL_DOC = False

def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        # try:
            if EVAL_DOC:
                l = l.strip().split(' ')
            else:
                l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            if  l[2][0] == "D":
                qids_to_relevant_passageids[qid].append(int(l[2][1:]))
            else:
                qids_to_relevant_passageids[qid].append(int(l[2]))
        # except:
        #     raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if l[1][0] == "D":
                pid = int(l[1][1:])
            else:
                pid = int(l[1])
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_passages:
                pass    
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1]=pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages
                
def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    
    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages

def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def mrr_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    
    MRR_li=[]
    for MaxMRRRank in MaxMRRRank_li:
        MRR = 0
        ranking = []
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                for i in range(0,MaxMRRRank):
                    if candidate_pid[i] in target_pid:
                        MRR += 1/(i + 1)
                        ranking.pop()
                        ranking.append(i+1)
                        break

        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
        
        MRR = MRR/len(qids_to_relevant_passageids)
        MRR_li.append(MRR)
    return MRR_li


def recall_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    recall_li=[]
    qry_number=len(qids_to_relevant_passageids)
    for MaxRECALLRank in MaxRECALLRank_li:
        recall=0
        for qid in qids_to_relevant_passageids:
            target = qids_to_relevant_passageids[qid][0]
            if target in qids_to_ranked_candidate_passages[qid][:MaxRECALLRank]:
                recall+=1
        recall/=qry_number
        recall_li.append(recall)
    return recall_li

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    # # 让起始下标一致
    # new_qids_to_ranked_candidate_passages={}
    # if min(qids_to_relevant_passageids)!=min(qids_to_ranked_candidate_passages):
    #     for qid in qids_to_ranked_candidate_passages:
    #         new_qids_to_ranked_candidate_passages[qid+1]=qids_to_ranked_candidate_passages[qid]
    # qids_to_ranked_candidate_passages=new_qids_to_ranked_candidate_passages


    MRR_li=mrr_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
    for i,MaxMRRRank in enumerate(MaxMRRRank_li):
        all_scores[f'MRR @{MaxMRRRank}'] = MRR_li[i]
    RECALL_li=recall_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
    for i,MaxRECALLRank in enumerate(MaxRECALLRank_li):
        all_scores[f'RECALL @{MaxRECALLRank}'] = RECALL_li[i]
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores
                
def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """
    print("Eval Started")
    path_to_reference = args.path_to_reference
    path_to_candidate = args.path_to_candidate
    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('#####################')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#####################')
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_reference",
        type=str,
        # default="./drhard//data/doc_subset/preprocess/query/train-qrel.tsv"
        default="./drhard/data/doc_subset/preprocess_multimodal_new/query/dev-qrel.tsv"
    )
    parser.add_argument(
        "--path_to_candidate",
        type=str,
        default="./drhard/data/evaluate/doc_subset/inbatch_train/leaf/bert/3e-05/dev.rank.tsv",
        # default="./drhard/data/evaluate/doc_subset/inbatch_train/pure_text/bert/3e-05/dev.rank.tsv"
    )
    args=parser.parse_args()
    main()
    # root="./drhard//ceshi1/doc_subset/evaluate/without_neg_3e-5_lamb_myloss_bs128/"
    # path_li=sorted(os.listdir(root))
    # for file in path_li:
    #     args.path_to_candidate=root+file+"/dev.rank.tsv"
    #     print(args.path_to_candidate)
    #     if os.path.exists(args.path_to_candidate):
    #         main()





    