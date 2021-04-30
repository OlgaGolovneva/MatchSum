import os
import argparse
from os.path import join, exists
import subprocess as sp
import json
import tempfile
import multiprocessing as mp
from time import time
from datetime import timedelta
import queue
import logging
import random
from itertools import combinations

from cytoolz import curry
from pyrouge.utils import log
from pyrouge import Rouge155

from transformers import BertTokenizer, RobertaTokenizer

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

MAX_LEN = 512

_ROUGE_PATH = '/Users/Olga/Projects/git/rouge/tools/ROUGE-1.5.5'
temp_path = './temp' # path to store some temporary files

original_data, sent_ids = [], []


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_rouge(path, dec):
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    dec_dir = join(path, 'decode')
    ref_dir = join(path, 'reference')

    with open(join(dec_dir, '0.dec'), 'w') as f:
        for sentence in dec:
            print(sentence, file=f)

    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
            + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
            + cmd
            + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

        line = output.split('\n')
        rouge1 = float(line[3].split(' ')[3])
        rouge2 = float(line[7].split(' ')[3])
        rougel = float(line[11].split(' ')[3])
    return (rouge1 + rouge2 + rougel) / 3


def sentence_bert_embedding(sentences):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = sbert_model.encode(sentences)
    return sentence_embeddings


def get_distances(vectors):
    '''
    Calculate matrix of cosine distances
    :param vectors: input vectors, list of lists of floats, size n
    :return: nxn distance matrix
    '''

    distance = [[0]*len(vectors) for i in range(len(vectors))]
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors), 1):
            distance[i][j] = distance[j][i] = cosine(vectors[i], vectors[j])

    return distance


def most_distant(pool, candidates, dist_matr):
    '''
    Iterate over all candidates and select the most outmost from the pool
    based on min distance
    :param pool: already selected vector indices
    :param candidates: candidate indices to be added to the pool
    :param dist_matr: distances between vectors
    :return:
    '''
    max_dist = 0
    most_dist = []
    for candidate in candidates:
        min = -1
        for ex_pont in pool:
            if min == -1 or min > dist_matr[candidate][ex_pont]:
                min = dist_matr[candidate][ex_pont]
        if min > max_dist:
            max_dist = min
            most_dist = candidate
    return most_dist


def outmost_vectors(vectors, count, distances):
    starting_point = random.randint(0, len(vectors) - 1)
    leftover = [i for i in range(len(vectors))]
    output = [starting_point]
    if starting_point not in leftover:
        print(starting_point)
    leftover.remove(starting_point)
    while len(output) < count:
        next_point = most_distant(output, leftover, distances)
        output.append(next_point)
        if next_point not in leftover:
            print(next_point)
        leftover.remove(next_point)

    return output


def group_by_similarity(sent_candidates, min_cand, max_cand, number_of_iterations=5):
    '''
    Given candidates, select subsets of length between min_cand and max_cand
    with most dissimilar utterances
    :param sent_candidates: all candidate sentences
    :param min_cand: min number of sentences in candidate
    :param max_cand: max number of sentences in candidate
    :param number_of_iterations: for each targeted number f sentences, how much time generate subset
    :return: list of lists with subsets of utternaces
    '''

    embeddings = sentence_bert_embedding(sent_candidates)
    distance_matrix = get_distances(embeddings)

    # collect indices of outmost vectors
    output = []

    for target in range(min_cand, max_cand + 1, 1):
        for iter in range(number_of_iterations):
            indices = outmost_vectors(embeddings, target, distance_matrix)
            sentences = [sent_candidates[i] for i in indices]
            output.append(sentences)

    return output


@curry
def get_candidates(tokenizer, cls, sep_id, idx):

    idx_path = join(temp_path, str(idx))
    
    # create some temporary files to calculate ROUGE
    sp.call('mkdir ' + idx_path, shell=True)
    sp.call('mkdir ' + join(idx_path, 'decode'), shell=True)
    sp.call('mkdir ' + join(idx_path, 'reference'), shell=True)
    
    # load data
    data = {}
    data['text'] = original_data[idx]['text']
    data['summary'] = original_data[idx]['summary']
    
    # write reference summary to temporary files
    ref_dir = join(idx_path, 'reference')
    with open(join(ref_dir, '0.ref'), 'w') as f:
        for sentence in data['summary']:
            print(sentence, file=f)

    # get candidate summaries
    # here is for CNN/DM: truncate each document into the 50 most important sentences (using BertExt),
    # then select between 20 and 40 sentences, each with 5 random initialization, to form a candidate summary,
    # so there are (40-20)*5=100 candidate summaries.
    # if you want to process other datasets, you may need to adjust these numbers according to specific situation.
    sent_id = sent_ids[idx]['sent_id'][:50]

    sentences = group_by_similarity([data['text'][index] for index in sent_id], 20, 40)
    
    # get ROUGE score for each candidate summary and sort them in descending order
    score = []
    for sent_set in sentences:
        score.append((sent_set, get_rouge(idx_path, sent_set)))
    score.sort(key=lambda x: x[1], reverse=True)
    
    # write candidate indices and score
    # tokenize and get candidate_id
    candidate_summary = []
    data['ext_idx'] = sent_id
    data['score'] = []
    data['indices'] = []
    for i, R in score:
        data_ind = []
        data['score'].append(R)
        cur_summary = [cls]
        for sentence in i:
            cur_summary += sentence.split()
            data_ind.append(data['text'].index(sentence))
        cur_summary = cur_summary[:MAX_LEN]
        cur_summary = ' '.join(cur_summary)
        candidate_summary.append(cur_summary)
        data['indices'].append(data_ind)
    
    data['candidate_id'] = []
    for summary in candidate_summary:
        token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
        token_ids += sep_id
        data['candidate_id'].append(token_ids)
    
    # tokenize and get text_id
    text = [cls]
    for sent in data['text']:
        text += sent.split()
    text = text[:MAX_LEN]
    text = ' '.join(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['text_id'] = token_ids
    
    # tokenize and get summary_id
    summary = [cls]
    for sent in data['summary']:
        summary += sent.split()
    summary = summary[:MAX_LEN]
    summary = ' '.join(summary)
    token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['summary_id'] = token_ids
    
    # write processed data to temporary file
    processed_path = join(temp_path, 'processed')
    with open(join(processed_path, '{}.json'.format(idx)), 'w') as f:
        json.dump(data, f, indent=4) 
    
    sp.call('rm -r ' + idx_path, shell=True)


def get_candidates_mp(args):
    
    # choose tokenizer
    if args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls, sep = '[CLS]', '[SEP]'
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        cls, sep = '<s>', '</s>'
    sep_id = tokenizer.encode(sep, add_special_tokens=False)

    # load original data and indices
    global original_data, sent_ids
    original_data = load_jsonl(args.data_path)
    sent_ids = load_jsonl(args.index_path)
    n_files = len(original_data)
    assert len(sent_ids) == len(original_data)
    print('total {} documents'.format(n_files))
    os.makedirs(temp_path)
    processed_path = join(temp_path, 'processed')
    os.makedirs(processed_path)

    # use multi-processing to get candidate summaries
    start = time()
    print('start getting candidates with multi-processing !!!')
    
    with mp.Pool() as pool:
        list(pool.imap_unordered(get_candidates(tokenizer, cls, sep_id), range(n_files), chunksize=64))
    
    print('finished in {}'.format(timedelta(seconds=time()-start)))
    
    # write processed data
    print('start writing {} files'.format(n_files))
    for i in range(n_files):
        with open(join(processed_path, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        with open(args.write_path, 'a') as f:
            print(json.dumps(data), file=f)
    
    os.system('rm -r {}'.format(temp_path))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )
    parser.add_argument('--tokenizer', type=str, required=True,
        help='BERT/RoBERTa')
    parser.add_argument('--data_path', type=str, required=True,
        help='path to the original dataset, the original dataset should contain text and summary')
    parser.add_argument('--index_path', type=str, required=True,
        help='indices of the remaining sentences of the truncated document')
    parser.add_argument('--write_path', type=str, required=True,
        help='path to store the processed dataset')

    args = parser.parse_args()
    assert args.tokenizer in ['bert', 'roberta']
    assert exists(args.data_path)
    assert exists(args.index_path)

    get_candidates_mp(args)
