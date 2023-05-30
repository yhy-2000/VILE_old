from scipy import stats
import collections
import argparse

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = {}
        for l in f:
            l = l.strip().split()
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            if l[2][0] == "D":
                qids_to_relevant_passageids[qid].append(int(l[2][1:]))
            else:
                qids_to_relevant_passageids[qid].append(int(l[2]))
        return qids_to_relevant_passageids

def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    
    with open(path_to_candidate,'r') as f:
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


def pair_wise_ttest(model_scores,baseline_scores):
    # 模型性能得分和基准线得分

    # 执行配对样本t-test
    t_statistic, p_value = stats.ttest_rel(model_scores, baseline_scores)

    # 判断模型是否显著优于所有基准线
    print("p_value: ",p_value)
    if p_value < 0.05:
        print("The model outperforms all baselines significantly.")
    else:
        print("The model does not outperform all baselines significantly.")


# 计算每一个查询的RR值
def mrr_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MaxMRRRank=10):
    MRR_li = []
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            ok=0
            for i in range(0,MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR_li.append(1/(i + 1))
                    ranking.pop()
                    ranking.append(i+1)
                    ok=1
                    break       
            if not ok:
                MRR_li.append(0)

    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    return MRR_li
 

def recall_calculate(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,MaxRECALLRank=10):
    qry_number=len(qids_to_relevant_passageids)
    recall_li=[]
    for qid in qids_to_relevant_passageids:
        target = qids_to_relevant_passageids[qid][0]
        if target in qids_to_ranked_candidate_passages[qid][:MaxRECALLRank]:
            recall_li.append(1)
        else:
            recall_li.append(0)
    return recall_li

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model_result_path",
        type=str,
        default="./drhard/data/evaluate/doc_subset/leaf/leaf/retromae/checkpoint-8649/dev.rank.tsv"
    )
    parser.add_argument(
        "--baseline_result_path",
        type=str,
        default="./drhard/data/evaluate/doc_subset/pure_text/pure_text/retromae/dev.rank.tsv",
    )
    parser.add_argument(
        "--qrel_path",
        type=str,
        default="./drhard/data/doc_subset/preprocess_multimodal_new/query/dev-qrel.tsv",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mrr",
        help="choose from [mrr,recall]"
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=10,
        help="choose from [1,10,100]"
    )
    for metric in ["mrr","recall"]:
        for truncate in [10,100]:
            args=parser.parse_args()
            qids_to_relevant_passageids=load_reference(args.qrel_path)
            model_result_rank=load_candidate(args.model_result_path)
            baseline_result_rank=load_candidate(args.baseline_result_path)
            if args.metric=="mrr":
                model_scores=mrr_calculate(qids_to_relevant_passageids, model_result_rank, MaxMRRRank=args.truncate)
                baseline_scores=mrr_calculate(qids_to_relevant_passageids, baseline_result_rank, MaxMRRRank=args.truncate)
            elif args.metric=="recall":
                model_scores=recall_calculate(qids_to_relevant_passageids, model_result_rank, MaxMRRRank=args.truncate)
                baseline_scores=recall_calculate(qids_to_relevant_passageids, baseline_result_rank, MaxMRRRank=args.truncate)
            
            pair_wise_ttest(model_scores,baseline_scores)



