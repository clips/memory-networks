import functools
import json
import os
import re
import subprocess
from collections import Counter
from functools import reduce
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch
from sklearn import model_selection
from sklearn.metrics import jaccard_similarity_score
from torch import nn
from torch.autograd import Variable

"""
The original code seems to be largely taken from https://github.com/siyuanzhao/key-value-memory-networks
"""

long_tensor_type = torch.LongTensor
float_tensor_type = torch.FloatTensor

if (torch.cuda.is_available()):
    long_tensor_type = torch.cuda.LongTensor
    float_tensor_type = torch.cuda.FloatTensor

DATA_KEY = "data"
VERSION_KEY = "version"
DOC_KEY = "document"
QAS_KEY = "qas"
ANS_KEY = "answers"
TXT_KEY = "text"  # the text part of the answer
ORIG_KEY = "origin"
ID_KEY = "id"
TITLE_KEY = "title"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
QUERY_KEY = "query"
CUI_KEY = "cui"
SEMTYPE_KEY = "sem_type"

PLACEHOLDER_KEY = "@placeholder"
SYMB_BEGIN = "@begin"
SYMB_END = "@end"


def process_data_clicr(args, log):
    data, val_data, test_data, vocab = load_data_clicr(args.data_dir, args.ent_setup, log, args.max_n_load)

    '''
    clicr data is of the form:
    [
        (
            [
                ['passage_w1', 'passage_w2', ...], 
                ['passage_w1', 'passage_w2', ...],
                ...
            ], 
            ['q_w1', 'q_w2', ...], 
            ['answer'],
            [
                ['cand_answer1'],
                ['cand_answer2']
            ],
            cloze_start_id,
            'query_id'
        ),
        .
        .
        .
        ()
    ]
    '''
    memory_size, sentence_size, vocab_size, word_idx, output_size, output_idx = calculate_parameter_values_clicr(data=data, debug=args.debug,
                                                                                        memory_size=args.memory_size,
                                                                                        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))
        log.info("Output Size: {}".format(output_size))

    return data, val_data, test_data, sentence_size, vocab_size, memory_size, word_idx, output_size, output_idx

def process_data_clicr_win(args, log):
    data, val_data, test_data, vocab = load_data_clicr_win(args.data_dir, args.ent_setup, log, args.max_n_load, args.win_size_kv, args.exclude_unseen_ans, args.max_vocab_size)

    '''
    clicr data is of the form:
    [
        (
            [
                ['passage_w1', 'passage_w2', ...], 
                ['passage_w1', 'passage_w2', ...],
                ...
            ], 
            ['q_w1', 'q_w2', ...], 
            ['answer'],
            [
                ['cand_answer1'],
                ['cand_answer2']
            ],
            cloze_start_id,
            'query_id'
        ),
        .
        .
        .
        ()
    ]
    '''
    memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values_clicr_win(data=data, debug=args.debug,
                                                                                        memory_size=args.memory_size,
                                                                                        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))

    return data, val_data, test_data, sentence_size, vocab_size, memory_size, word_idx


def process_data_cbt_win(args, log):
    data, val_data, test_data, vocab = load_data_cbt_win(args.data_dir, args.ent_setup, log, args.max_n_load, args.win_size_kv, args.dataset_part, args.exclude_unseen_ans)

    '''
    cbt win data is of the form:
    [
        (
            [
                ['win_i-n', ..., 'win_i', ..., win_i+n], 
                ['win_i-n', ..., 'win_i', ..., win_i+n],
                ...
            ], 
            ['q_i-n', ..., 'q_i', ..., 'q_i+n'], 
            ['answer'],
            [
                ['cand_answer1'],
                ['cand_answer2']
            ],
            cloze_start_id,
            'query_id'
        ),
        .
        .
        .
        ()
    ]
    '''
    memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values_cbt_win(data=data, debug=args.debug,
                                                                                        memory_size=args.memory_size,
                                                                                        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))
        #log.info("Output Size: {}".format(output_size))

    return data, val_data, test_data, sentence_size, vocab_size, memory_size, word_idx


def process_data_clicr_kv(args, log):
    data, val_data, test_data, vocab = load_data_clicr_kv(args.data_dir, args.ent_setup, log, args.win_size_kv, args.max_n_load)

    '''
    clicr data is of the form:
    [
        (
            (
                [['k1', 'k2', ...], ...] 
                [['v1', 'v2', ...], ...]
            ), 
            ['q_w1', 'q_w2', ...], 
            ['answer'],
            [
                ['cand_answer1'],
                ['cand_answer2']
            ],
            cloze_start_id,
            'query_id'
        ),
        .
        .
        .
        ()
    ]
    '''
    memory_size, k_size, v_size, vocab_size, word_idx, output_size, output_idx = calculate_parameter_values_clicr_kv(data=data, debug=args.debug,
                                                                                        memory_size=args.memory_size,
                                                                                        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))
        log.info("Output Size: {}".format(output_size))

    return data, val_data, test_data, k_size, v_size, vocab_size, memory_size, word_idx, output_size, output_idx


def process_data_cbt_kv(args, log):
    data, val_data, test_data, vocab = load_data_clicr_kv(args.data_dir, args.ent_setup, log, args.win_size_kv, args.max_n_load)

    '''
    cbt data is of the form:
    [
        (
            (
                [['k1', 'k2', ...], ...] 
                [['v1', 'v2', ...], ...]
            ), 
            ['q_w1', 'q_w2', ...], 
            ['answer'],
            [
                ['cand_answer1'],
                ['cand_answer2']
            ],
            cloze_start_id,
            'query_id'
        ),
        .
        .
        .
        ()
    ]
    '''
    memory_size, k_size, v_size, vocab_size, word_idx, output_size, output_idx = calculate_parameter_values_clicr_kv(data=data, debug=args.debug,
                                                                                        memory_size=args.memory_size,
                                                                                        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))
        log.info("Output Size: {}".format(output_size))

    return data, val_data, test_data, k_size, v_size, vocab_size, memory_size, word_idx, output_size, output_idx


def process_data_kv(args, log):
    data, test_data, vocab = load_data_kv(args.data_dir, args.joint_training, args.task_number, args.win_size_kv)

    '''
    
    '''
    memory_size, k_size, v_size, vocab_size, word_idx = calculate_parameter_values_kv(
        data=data, debug=args.debug,
        memory_size=args.memory_size,
        vocab=vocab, log=log)
    if args.debug:
        log.info("Vocabulary Size: {}".format(vocab_size))

    return data, test_data, k_size, v_size, vocab_size, memory_size, word_idx

def load_clicr_win(fn, ent_setup="ent", remove_notfound=True, max_n_load=None, win_size=3):
    questions = []
    raw = load_json(fn)
    for c, datum in enumerate(raw[DATA_KEY]):
        doc_txt = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        # keys include values; we'll only be using keys here
        keys = prepare_win(doc_txt, win_size=win_size)  # n_words*d

        sents = []
        for sent in doc_txt.split("\n"):
            if sent:
                sents.append(to_entities(sent))
        document = " ".join(sents)

        for qa in datum[DOC_KEY][QAS_KEY]:
            doc_raw = document.split()
            query_id = qa[ID_KEY]
            query = qa[QUERY_KEY]
            query_win = prepare_q_for_kv(query, win_size=win_size)
            ans_raw = ""
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
            assert ans_raw
            if remove_notfound:  # should be always false for dev and test
                if ans_raw not in doc_raw:
                    found_umls = False
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "UMLS":
                            umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                            if umls_answer in doc_raw:
                                found_umls = True
                                ans_raw = umls_answer
                    if not found_umls:
                        continue
            cand_e = [w.lower() for w in doc_raw if w.startswith('@entity')]
            cand_raw = [[e] for e in cand_e]
            questions.append((keys, query_win, [ans_raw], cand_raw, None, query_id))
        if max_n_load is not None and c > max_n_load:
            break

    return questions



def load_clicr_kv(fn, ent_setup="ent", win_size=3, remove_notfound=True, max_n_load=None):
    questions = []
    raw = load_json(fn)
    for c, datum in enumerate(raw[DATA_KEY]):
        doc_txt = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        keys, values = prepare_kv(doc_txt, win_size=win_size)  # n_words*d
        assert len(keys) == len(values)

        sents = []
        for sent in doc_txt.split("\n"):
            if sent:
                sents.append(to_entities(sent))
        document = " ".join(sents)

        for qa in datum[DOC_KEY][QAS_KEY]:
            doc_raw = document.split()
            query_id = qa[ID_KEY]
            query = qa[QUERY_KEY]
            query_win = prepare_q_for_kv(query, win_size=win_size)
            ans_raw = ""
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
            assert ans_raw
            if remove_notfound:  # should be always false for dev and test
                if ans_raw not in doc_raw:
                    found_umls = False
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "UMLS":
                            umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                            if umls_answer in doc_raw:
                                found_umls = True
                                ans_raw = umls_answer
                    if not found_umls:
                        continue
            cand_e = [w.lower() for w in doc_raw if w.startswith('@entity')]
            cand_raw = [[e] for e in cand_e]
            questions.append(((keys, values), query_win, [ans_raw], cand_raw, None, query_id))
        if max_n_load is not None and c > max_n_load:
            break

    return questions


def load_clicr_kv_ent_only(fn, ent_setup="ent", win_size=3, remove_notfound=True, max_n_load=None):
    questions = []
    raw = load_json(fn)
    for c, datum in enumerate(raw[DATA_KEY]):
        doc_txt = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        keys, values = prepare_kv_ent_only(doc_txt, win_size=win_size)  # n_words*d
        assert len(keys) == len(values)

        sents = []
        for sent in doc_txt.split("\n"):
            if sent:
                sents.append(to_entities(sent))
        document = " ".join(sents)

        for qa in datum[DOC_KEY][QAS_KEY]:
            doc_raw = document.split()
            query_id = qa[ID_KEY]
            query = qa[QUERY_KEY]
            query_win = prepare_q_for_kv(query, win_size=win_size)
            ans_raw = ""
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
            assert ans_raw
            if remove_notfound:  # should be always false for dev and test
                if ans_raw not in doc_raw:
                    found_umls = False
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "UMLS":
                            umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                            if umls_answer in doc_raw:
                                found_umls = True
                                ans_raw = umls_answer
                    if not found_umls:
                        continue
            cand_e = [w.lower() for w in doc_raw if w.startswith('@entity')]
            cand_raw = [[e] for e in cand_e]
            questions.append(((keys, values), query_win, [ans_raw], cand_raw, None, query_id))
        if max_n_load is not None and c > max_n_load:
            break

    return questions



def prepare_q_for_kv(q, win_size=3):
    q_line = ""
    for line in q.split("\n"):
        if PLACEHOLDER_KEY in line:
            q_line = line
    assert q_line
    q_line_lst = remove_concept_marks(q_line).split()
    q_line = " ".join(q_line_lst)
    idx_start = q_line.find(PLACEHOLDER_KEY)
    idx_end = idx_start + len(PLACEHOLDER_KEY)
    if len(q_line) > idx_end:
        if q_line[idx_end] != " ":
            q_line = q_line[:idx_end] + " " + q_line[idx_end:]
    txt_left = q_line[:idx_start].rstrip().split()
    txt_right = q_line[idx_end:].lstrip().split()
    txt = txt_left + txt_right
    assert not len(txt) == len(q_line.strip().split()), q_line  # removed placeholder
    i = len(txt_left)
    window_start = max(0, i - win_size)
    window_end = min(len(txt), i + win_size)
    contexts = []
    # go over contexts
    for j in range(window_start, window_end):
        c = txt[j]
        contexts.append(c.lower())
    assert contexts

    return contexts


def prepare_kv_babi(text, win_size=3):
    """
    [["w1", "w2", ...],
    ["w9", "w10", ...]]
    """
    values = []
    keys = []  # n_words*(2*win_size)
    for sent in text:
        for c, w in enumerate(sent):
            left = sent[max(0, c-win_size):c]
            right = sent[c+1:c+1+win_size]
            contexts = left + right
            if not contexts:
                continue
            keys.append(contexts)
            values.append(w)
    assert len(values) > 0

    return keys, values


def prepare_kv(text, win_size=3):
    values = []
    keys = []  # n_words*(2*win_size)
    for line in text.split("\n"):
        idxs_start = [match.start() for match in re.finditer("BEG__", line)]
        idxs_end = [match.end() for match in re.finditer("__END", line)]
        for i_start, i_end in zip(idxs_start, idxs_end):
            concept = line[i_start + len("BEG__"):i_end - len("__END")]
            concept = "@entity" + concept.replace(" ", "_").lower()
            txt_left = line[:i_start].strip()
            lst_left = txt_left.split()
            txt_right = line[i_end:].strip()
            lst_right = txt_right.split()
            lst = lst_left + lst_right
            i = len(lst_left)
            window_start = max(0, i - win_size)
            window_end = min(len(lst), i + win_size)
            contexts = []
            # go over contexts
            for j in range(window_start, window_end):
                w = lst[j]
                w = remove_concept_marks(w)
                contexts.append(w.lower())
            if not contexts:
                continue
            values.append(concept)
            keys.append(contexts)

    assert len(values) > 0

    return keys, values


def prepare_win(text, win_size=3):
    values = []
    keys = []  # n_words*(2*win_size)
    for line in text.split("\n"):
        idxs_start = [match.start() for match in re.finditer("BEG__", line)]
        idxs_end = [match.end() for match in re.finditer("__END", line)]
        for i_start, i_end in zip(idxs_start, idxs_end):
            concept = line[i_start + len("BEG__"):i_end - len("__END")]
            concept = "@entity" + concept.replace(" ", "_").lower()
            txt_left = line[:i_start].strip()
            lst_left = txt_left.split()
            txt_right = line[i_end:].strip()
            lst_right = txt_right.split()
            lst = lst_left + lst_right
            i = len(lst_left)
            window_start = max(0, i - win_size)
            window_end = min(len(lst), i + win_size)
            contexts = []
            # go over contexts
            #for j in range(window_start, window_end):
            #    w = lst[j]
            #    w = remove_concept_marks(w)
            #    contexts.append(w.lower())
            #contexts.append(concept)
            for j in range(window_start, i):
                w = lst[j]
                w = remove_concept_marks(w)
                contexts.append(w.lower())
            contexts.append(concept)
            for j in range(i, window_end):
                w = lst[j]
                w = remove_concept_marks(w)
                contexts.append(w.lower())
            if not contexts:
                continue
            values.append(concept)
            keys.append(contexts)

    assert len(values) > 0
    assert len(keys) == len(values)
    return keys


def prepare_kv_ent_only(text, win_size=3):
    values = []
    keys = []  # n_words*(2*win_size)
    for line in text.split("\n"):
        idxs_start = [match.start() for match in re.finditer("BEG__", line)]
        idxs_end = [match.end() for match in re.finditer("__END", line)]
        for i_start, i_end in zip(idxs_start, idxs_end):
            concept = line[i_start + len("BEG__"):i_end - len("__END")]
            concept = "@entity" + concept.replace(" ", "_").lower()
            txt_left = line[:i_start].strip()
            lst_left = txt_left.split()
            txt_right = line[i_end:].strip()
            lst_right = txt_right.split()
            lst = lst_left + lst_right
            i = len(lst_left)
            window_start = max(0, i - win_size)
            window_end = min(len(lst), i + win_size)
            contexts = []
            # go over contexts
            for j in range(window_start, window_end):
                w = lst[j]
                w = remove_concept_marks(w)
                contexts.append(w.lower())
            if not contexts:
                continue
            values.append(concept)
            keys.append(contexts)

    assert len(values) > 0

    return keys, values


def remove_concept_marks(txt, marker1="BEG__", marker2="__END"):
    return txt.replace(marker1, "").replace(marker2, "")


def process_data(args, log):
    data, val_data, test_data, vocab = load_data(args.data_dir, args.joint_training, args.task_number)

    '''
    data is of the form:
    [
        (
            [
                ['mary', 'moved', 'to', 'the', 'bathroom'], 
                ['john', 'went', 'to', 'the', 'hallway']
            ], 
            ['where', 'is', 'mary'], 
            ['bathroom']
        ),
        .
        .
        .
        ()
    ]
    '''

    memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values(data=data, debug=args.debug,
                                                                                  memory_size=args.memory_size,
                                                                                  vocab=vocab, log=log)

    return data, val_data, test_data, sentence_size, vocab_size, memory_size, word_idx


def load_data_clicr(data_dir, ent_setup, log, max_n_load=None):
    #train_data, _ = load_clicr_ent_only(data_dir + "train1.0.json", ent_setup, max_n_load=max_n_load)
    train_data, _ = load_clicr(data_dir + "train1.0.json", ent_setup, max_n_load=max_n_load)
    val_data, _ = load_clicr(data_dir + "dev1.0.json", ent_setup, remove_notfound=False, max_n_load=max_n_load)
    test_data, _ = load_clicr(data_dir + "test1.0.json", ent_setup, remove_notfound=False, max_n_load=max_n_load)

    data = train_data + val_data + test_data  # TODO exclude test?

    vocab_set = set()
    for s, q, a, _, _, _ in data:
        vocab_set.update([w for sent in s for w in sent] + q + a)

    vocab = sorted(vocab_set)

    return train_data, val_data, test_data, vocab


def cbt_stats(train, test):
    ans_test = [i[2][0] for i in test]
    n_ans_test = len(ans_test)
    n_ans_types_test = len(set(ans_test))

    ans_train = [i[2][0] for i in train]
    n_ans_train = len(ans_train)
    n_ans_types_train = len(set(ans_train))

    n_train_ans_in_test = len(set(ans_train) & set(ans_test))

    print("test: n ans tok {} types {} ration {}".format(n_ans_test, n_ans_types_test, n_ans_types_test / n_ans_test))
    print("train: n ans tok {} types {} ration {}".format(n_ans_train, n_ans_types_train, n_ans_types_train / n_ans_train))
    print("n train ans in test {} / {} all test".format(n_train_ans_in_test, n_ans_types_test))


def prune_test(train_data, test_data):
    ans_train = {i[2][0] for i in train_data}
    print("ans train len: {}".format(len(ans_train)))

    new_test_data = []
    for i in test_data:
        if i[2][0] in ans_train:
            new_test_data.append(i)

    return new_test_data

def load_data_clicr_win(data_dir, ent_setup, log, max_n_load=None, win_size=3, exclude_unseen_ans=False, max_vocab_size=1e50):
    #train_data, _ = load_clicr_ent_only(data_dir + "train1.0.json", ent_setup, max_n_load=max_n_load)

    #train_data_ne, _ = load_cbt_win(data_dir + "cbtest_NE_train.txt", ent_setup, max_n_load=max_n_load, win_size=win_size)
    #train_data_cn, _ = load_cbt_win(data_dir + "cbtest_CN_train.txt", ent_setup, max_n_load=max_n_load,
    #                                win_size=win_size)
    #train_data_p, _ = load_cbt_win(data_dir + "cbtest_P_train.txt", ent_setup, max_n_load=max_n_load,
    #                                win_size=win_size)
    #train_data_v, _ = load_cbt_win(data_dir + "cbtest_V_train.txt", ent_setup, max_n_load=max_n_load,
    #                               win_size=win_size)
    #train_data = train_data_ne + train_data_cn + train_data_p + train_data_v
    #np.random.seed(1234)
    #np.random.shuffle(train_data)
    train_data = load_clicr_win(data_dir + "train1.0.json", ent_setup, remove_notfound=True, max_n_load=max_n_load, win_size=win_size)
    val_data = load_clicr_win(data_dir + "dev1.0.json", ent_setup, remove_notfound=False, max_n_load=max_n_load, win_size=win_size)
    test_data = load_clicr_win(data_dir + "test1.0.json", ent_setup, remove_notfound=False, max_n_load=max_n_load, win_size=win_size)

    if exclude_unseen_ans:
        test_data = prune_test(train_data, test_data)

    cbt_stats(train_data, test_data)
    data = train_data + val_data + test_data  # TODO exclude test?

    #vocab_set = set()
    #for s, q, a, _, _, _ in data:
    #    vocab_set.update([w for sent in s for w in sent] + q + a)
    vocab_cnt = Counter()
    for s, q, a, _, _, _ in data:
        vocab_cnt.update([w for sent in s for w in sent] + q + a)
    
    #vocab = sorted(vocab_set)
    vocab = sorted([w for w,f in vocab_cnt.most_common(max_vocab_size)] + ["_UNK_"])

    return train_data, val_data, test_data, vocab

def load_data_cbt_win(data_dir, ent_setup, log, max_n_load=None, win_size=3, dataset_part="NE", exclude_unseen_ans=False):
    #train_data, _ = load_clicr_ent_only(data_dir + "train1.0.json", ent_setup, max_n_load=max_n_load)

    #train_data_ne, _ = load_cbt_win(data_dir + "cbtest_NE_train.txt", ent_setup, max_n_load=max_n_load, win_size=win_size)
    #train_data_cn, _ = load_cbt_win(data_dir + "cbtest_CN_train.txt", ent_setup, max_n_load=max_n_load,
    #                                win_size=win_size)
    #train_data_p, _ = load_cbt_win(data_dir + "cbtest_P_train.txt", ent_setup, max_n_load=max_n_load,
    #                                win_size=win_size)
    #train_data_v, _ = load_cbt_win(data_dir + "cbtest_V_train.txt", ent_setup, max_n_load=max_n_load,
    #                               win_size=win_size)
    #train_data = train_data_ne + train_data_cn + train_data_p + train_data_v
    #np.random.seed(1234)
    #np.random.shuffle(train_data)
    train_data, _ = load_cbt_win(data_dir + "cbtest_{}_train.txt".format(dataset_part), ent_setup, max_n_load=max_n_load, win_size=win_size)
    val_data, _ = load_cbt_win(data_dir + "cbtest_{}_valid_2000ex.txt".format(dataset_part), ent_setup, remove_notfound=False, max_n_load=max_n_load, win_size=win_size)
    test_data, _ = load_cbt_win(data_dir + "cbtest_{}_test_2500ex.txt".format(dataset_part), ent_setup, remove_notfound=False, max_n_load=max_n_load, win_size=win_size)

    if exclude_unseen_ans:
        test_data = prune_test(train_data, test_data)

    cbt_stats(train_data, test_data)
    data = train_data + val_data + test_data  # TODO exclude test?

    vocab_set = set()
    for s, q, a, _, _, _ in data:
        vocab_set.update([w for sent in s for w in sent] + q + a)

    vocab = sorted(vocab_set)

    return train_data, val_data, test_data, vocab


def load_data_clicr_kv(data_dir, ent_setup, log, win_size=3, max_n_load=None):
    #train_data, _ = load_clicr_ent_only(data_dir + "train1.0.json", ent_setup, max_n_load=max_n_load)
    train_data = load_clicr_kv(data_dir + "train1.0.json", win_size=win_size, ent_setup=ent_setup, max_n_load=max_n_load)
    val_data = load_clicr_kv(data_dir + "dev1.0.json", win_size=win_size, ent_setup=ent_setup, remove_notfound=False, max_n_load=max_n_load)
    test_data = load_clicr_kv(data_dir + "test1.0.json", win_size=win_size, ent_setup=ent_setup, remove_notfound=False, max_n_load=max_n_load)

    data = train_data + val_data + test_data  # TODO exclude test?

    vocab_set = set()
    for (k,v), q, a, _, _, _ in data:
        vocab_set.update([w for sent in k for w in sent] + v + q + a)

    vocab = sorted(vocab_set)

    return train_data, val_data, test_data, vocab


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def get_q_ids_clicr(fn):
    q_ids = set()
    dataset = load_json(fn)
    data = dataset[DATA_KEY]
    for datum in data:
        for qa in datum[DOC_KEY][QAS_KEY]:
            q_ids.add(qa[ID_KEY])

    return q_ids


def document_instance(context, title, qas):
    return {"context": context, "title": title, "qas": qas}


def dataset_instance(version, data):
    return {"version": version, "data": data}


def datum_instance(document, source):
        return {"document": document, "source": source}


def intersect_on_ids(dataset, predictions):
    """
    Reduce data to exclude all qa ids but those in  predictions.
    """
    new_data = []

    for datum in dataset[DATA_KEY]:
        qas = []
        for qa in datum[DOC_KEY][QAS_KEY]:
            if qa[ID_KEY] in predictions:
                qas.append(qa)
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    return dataset_instance(dataset[VERSION_KEY], new_data)


def remove_missing_preds(fn, predictions):
    dataset = load_json(fn)
    new_dataset = intersect_on_ids(dataset, predictions)

    return new_dataset


def load_clicr(fn, ent_setup="ent", remove_notfound=True, max_n_load=None):
    questions = []
    raw = load_json(fn)
    relabeling_dicts = {}
    for c, datum in enumerate(raw[DATA_KEY]):
        sents = []
        for sent in (datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).split("\n"):
            if sent:
                sents.append(to_entities(sent).lower())
        document = " ".join(sents)
        for qa in datum[DOC_KEY][QAS_KEY]:
            if ent_setup in ["ent-anonym", "ent"]:
                doc_raw = document.split()
                question = to_entities(qa[QUERY_KEY]).lower()
                qry_id = qa[ID_KEY]
                assert question
                ans_raw = ""
                for ans in qa[ANS_KEY]:
                    if ans[ORIG_KEY] == "dataset":
                        ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                assert ans_raw
                if remove_notfound:  # should be always false for dev and test
                    if ans_raw not in doc_raw:
                        found_umls = False
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "UMLS":
                                umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                if umls_answer in doc_raw:
                                    found_umls = True
                                    ans_raw = umls_answer
                        if not found_umls:
                            continue
                qry_raw = question.split()
                if ent_setup == "ent-anonym":
                    entity_dict = {}
                    entity_id = 0
                    lst = doc_raw + qry_raw
                    lst.append(ans_raw)
                    for word in lst:
                        if (word.startswith('@entity')) and (word not in entity_dict):
                            entity_dict[word] = '@entity' + str(entity_id)
                            entity_id += 1
                    qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                    doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                    ans_raw = entity_dict[ans_raw]
                    inv_entity_dict = {ent_id: ent_ans for ent_ans, ent_id in entity_dict.items()}
                    assert len(entity_dict) == len(inv_entity_dict)
                    relabeling_dicts[qa[ID_KEY]] = inv_entity_dict

                cand_e = [w for w in doc_raw if w.startswith('@entity')]
                cand_raw = [[e] for e in cand_e]
                # wrap the query with special symbols
                qry_raw.insert(0, SYMB_BEGIN)
                qry_raw.append(SYMB_END)
                try:
                    cloze = qry_raw.index('@placeholder')
                except ValueError:
                    print('@placeholder not found in ', qry_raw, '. Fixing...')
                    at = qry_raw.index('@')
                    qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                    cloze = qry_raw.index('@placeholder')

                questions.append(([sent.split() for sent in sents], qry_raw, [ans_raw], cand_raw, cloze, qry_id))

            elif ent_setup == "no-ent":
                # collect candidate ents using @entity marks
                cand_e = [w for w in
                          to_entities(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).lower().split() if
                          w.startswith('@entity')]
                cand_raw = [e[len("@entity"):].split("_") for e in cand_e]
                # document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                # document = document.lower()
                # doc_raw = document.split()
                # sents = document.split("\n")
                sents = []
                for sent in (datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).split("\n"):
                    if sent:
                        sents.append(remove_entity_marks(sent).lower())
                document = " ".join(sents)
                doc_raw = document.split()
                question = remove_entity_marks(qa[QUERY_KEY]).lower()
                qry_id = qa[ID_KEY]
                assert question
                qry_raw = question.split()
                ans_raw = ""
                for ans in qa[ANS_KEY]:
                    if ans[ORIG_KEY] == "dataset":
                        ans_raw = ans[TXT_KEY].lower()
                assert ans_raw
                if remove_notfound:
                    if ans_raw not in doc_raw:
                        found_umls = False
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "UMLS":
                                umls_answer = ans[TXT_KEY].lower()
                                if umls_answer in doc_raw:
                                    found_umls = True
                                    ans_raw = umls_answer
                        if not found_umls:
                            continue

                relabeling_dicts[qa[ID_KEY]] = None
                # wrap the query with special symbols
                qry_raw.insert(0, SYMB_BEGIN)
                qry_raw.append(SYMB_END)
                try:
                    cloze = qry_raw.index('@placeholder')
                except ValueError:
                    print('@placeholder not found in ', qry_raw, '. Fixing...')
                    at = qry_raw.index('@')
                    qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                    cloze = qry_raw.index('@placeholder')

                questions.append(([sent.split() for sent in sents], qry_raw, [ans_raw], cand_raw, cloze, qry_id))
            else:
                raise ValueError
        if max_n_load is not None and c > max_n_load:
            break
    return questions, relabeling_dicts


def load_clicr_ent_only(fn, ent_setup="ent", remove_notfound=True, max_n_load=None):
    """
    only entities as text
    """
    questions = []
    raw = load_json(fn)
    relabeling_dicts = {}
    for c, datum in enumerate(raw[DATA_KEY]):
        sents = []
        for sent in (datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).split("\n"):
            if sent:
                ent_sent = " ".join([w for w in to_entities(sent).lower().split(" ") if w.startswith("@entity")])
                if ent_sent:
                    sents.append(ent_sent)
        document = " ".join(sents)
        for qa in datum[DOC_KEY][QAS_KEY]:
            if ent_setup in ["ent"]:
                doc_raw = document.split()
                question = " ".join([w for w in to_entities(qa[QUERY_KEY]).lower().split(" ") if w.startswith("@entity") or w.startswith("@placeholder")])
                if not question:
                    continue
                qry_id = qa[ID_KEY]
                assert question
                ans_raw = ""
                for ans in qa[ANS_KEY]:
                    if ans[ORIG_KEY] == "dataset":
                        ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                assert ans_raw
                if remove_notfound:  # should be always false for dev and test
                    if ans_raw not in doc_raw:
                        found_umls = False
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "UMLS":
                                umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                if umls_answer in doc_raw:
                                    found_umls = True
                                    ans_raw = umls_answer
                        if not found_umls:
                            continue
                qry_raw = question.split()

                cand_e = [w for w in doc_raw if w.startswith('@entity')]
                cand_raw = [[e] for e in cand_e]
                # wrap the query with special symbols
                qry_raw.insert(0, SYMB_BEGIN)
                qry_raw.append(SYMB_END)
                try:
                    cloze = qry_raw.index('@placeholder')
                except ValueError:
                    print('@placeholder not found in ', qry_raw, '. Fixing...')
                    at = qry_raw.index('@')
                    qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                    cloze = qry_raw.index('@placeholder')

                questions.append(([sent.split() for sent in sents], qry_raw, [ans_raw], cand_raw, cloze, qry_id))
            else:
                raise ValueError
        if max_n_load is not None and c > max_n_load:
            break
    return questions, relabeling_dicts


def process_inst_cbt(i):
    ls = i.strip().split("\n")
    if len(ls) != 21:
        return None
    else:
        sents = []
        q = []
        a = ""
        cands = []
        for c, l in enumerate(ls):
            n, s = l.split(" ", 1)
            n = int(n)
            assert n == c+1
            if c < 20:
                sents.append(s)
            elif c == 20:
                q, a, _, cands = s.split("\t")
                cands = cands.split("|")
        return (sents, q, a, cands)


def read_cbt(fn, lowercase=True):
    def to_lower(s, low):
        return s.lower() if low else s

    proc_insts = []
    with open(fn) as f:
        raw_insts = to_lower(f.read(), lowercase).split("\n\n")
        for i in tqdm(raw_insts):
            inst = process_inst_cbt(i)
            if inst is not None:
                proc_insts.append(inst)
    print("\nn inst {}: {}".format(fn, len(proc_insts)))
    return proc_insts


def get_win(sent, cands, win_size=3, include_cand=True):
    """
    :param sent: a list of words
    :param cands: a set of cands
    """
    for c, w in enumerate(sent):
        if w in cands:
            left = sent[max(0, c - win_size):c]
            right = sent[c + 1:c + 1 + win_size]
            win = left + [w] + right if include_cand else left + right

            yield win, w


def load_cbt_win(fn, ent_setup="ent", remove_notfound=True, max_n_load=None, win_size=3, include_cand=True):
    questions = []
    insts = read_cbt(fn)
    relabeling_dicts = {}
    #max_mem_size = 0
    for c, inst in enumerate(insts):
        sents, q, a, cands = inst
        wins = [(win, w) for s in sents for win, w in get_win(s.split(), set(cands), win_size=win_size, include_cand=include_cand)]
        #if len(wins)> max_mem_size:
        #    max_mem_size = len(wins)
        q_win = next(get_win(q.split(), {"xxxxx"}, win_size=win_size, include_cand=include_cand))
        cloze = q.index("xxxxx")
        if include_cand:
            wins = [win for win, w in wins]
            q_win = q_win[0]
        questions.append((wins, q_win, [a], [[c] for c in cands], cloze, c))
        if max_n_load is not None and c > max_n_load:
            break
    #print("maximum memory size in {}: {}".format(os.path.basename(fn), max_mem_size))
    return questions, relabeling_dicts


def remove_entity_marks(txt):
    return txt.replace("BEG__", "").replace("__END", "")


def to_entities(text):
    """
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a single entity @entityw1_w2_w3.
    """
    word_list = []
    inside = False
    concept = None
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = [w_stripped.split("_")[2]]
            word_list.append("@entity" + "_".join(concept))
            if inside:  # something went wrong, leave as is
                print("Inconsistent markup.")
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            if not inside:
                # add incorrectly parsed concept, but without entity marking
                word_list.append(w_stripped.rsplit("_", 2)[0])
                continue
            assert inside
            concept.append(w_stripped.rsplit("_", 2)[0])
            word_list.append("@entity" + "_".join(concept))
            inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                word_list.append(w_stripped)
    if concept and inside:
        # add incorrectly parsed concept, but without entity marking
        word_list.extend(concept)

    return " ".join(word_list)


def load_data(data_dir, joint_training, task_number):
    if (joint_training == 0):
        start_task = task_number
        end_task = task_number
    else:
        start_task = 1
        end_task = 20

    train_data = []
    test_data = []

    while start_task <= end_task:
        task_train, task_test = load_task(data_dir, start_task)
        train_data += task_train
        test_data += task_test
        start_task += 1

    np.random.shuffle(train_data)
    val_size = int(len(train_data)*0.1)
    val_data, train_data = train_data[:val_size], train_data[val_size:]
    data = train_data + val_data + test_data

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a, _, _, _ in data)))

    return train_data, val_data, test_data, vocab


def load_data_kv(data_dir, joint_training, task_number, win_size):
    if (joint_training == 0):
        start_task = task_number
        end_task = task_number
    else:
        start_task = 1
        end_task = 20

    train_data = []
    test_data = []

    while start_task <= end_task:
        task_train, task_test = load_task(data_dir, start_task, kv=True, win_size=win_size)
        train_data += task_train
        test_data += task_test
        start_task += 1

    data = train_data + test_data

    #vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for (k, v), q, a in data)))
    vocab_set = set()
    for (k, v), q, a in data:
        vocab_set.update([w for sent in k for w in sent] + v + q + a)

    vocab = sorted(vocab_set)

    return train_data, test_data, vocab


def load_task(data_dir, task_id, only_supporting=False, kv=False, win_size=3):
    '''
    Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting, kv=kv, win_size=win_size)
    test_data = get_stories(test_file, only_supporting, kv=kv, win_size=win_size)
    return train_data, test_data


def get_stories(f, only_supporting=False, kv=False, win_size=3):
    '''
    Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting, kv=kv, win_size=win_size)


def parse_stories(lines, only_supporting=False, kv=False, win_size=3):
    '''
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split(''))
                if kv:
                    raise NotImplementedError
                else:
                    substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                if kv:
                    substory = prepare_kv_babi(story, win_size=win_size)
                else:
                    substory = [x for x in story if x]

            data.append((substory, q, a, None, None, None))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def tokenize(sent):
    '''
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def calculate_parameter_values(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (s for s, _, _, _, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _, _, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _, _ in data)))
    query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position
    if debug:
        log.info("Longest sentence length: {}".format(sentence_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Average memory size: {}".format(memory_size))
    return memory_size, sentence_size, vocab_size, word_idx


def calculate_parameter_values_kv(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (k for (k,v), _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (k for (k,v), _, _ in data)))))
    k_size = max(map(len, chain.from_iterable(k for (k,v), _, _ in data)))
    v_size = None
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    if debug:
        log.info("Longest key length: {}".format(k_size))
        log.info("Longest value length: {}".format(v_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Average memory size: {}".format(memory_size))
    return memory_size, k_size, v_size, vocab_size, word_idx


def calculate_parameter_values_clicr(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    output_idx = dict()
    i = 0
    for w in vocab:
        if w.startswith("@entity"):
            output_idx[w] = i
            i += 1
    max_story_size = max(map(len, (s for s, _, _, _, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _, _, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _, _ in data)))
    query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    output_size = len(output_idx)
    sentence_size = max(query_size, sentence_size)  # for the position
    if debug is True:
        log.info("Longest sentence length: {}".format(sentence_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Average memory size: {}".format(memory_size))
    return memory_size, sentence_size, vocab_size, word_idx, output_size, output_idx


def calculate_parameter_values_clicr_win(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (s for s, _, _, _, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _, _, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _, _ in data)))
    query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position
    if debug is True:
        log.info("Longest sentence length: {}".format(sentence_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Average memory size: {}".format(memory_size))
    return memory_size, sentence_size, vocab_size, word_idx


def calculate_parameter_values_cbt_win(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    #output_idx = dict()
    #i = 0
    #for w in vocab:
    #    if w.startswith("@entity"):
    #        output_idx[w] = i
    #        i += 1
    max_story_size = max(map(len, (s for s, _, _, _, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _, _, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _, _ in data)))
    query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    #output_size = len(output_idx)
    sentence_size = max(query_size, sentence_size)  # for the position
    if debug is True:
        log.info("Longest sentence length: {}".format(sentence_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Average memory size: {}".format(memory_size))
    return memory_size, sentence_size, vocab_size, word_idx


def calculate_parameter_values_clicr_kv(data, debug, memory_size, vocab, log):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    output_idx = dict()
    i = 0
    for w in vocab:
        if w.startswith("@entity"):
            output_idx[w] = i
            i += 1
    max_story_size = max(map(len, (k for (k,v), _, _, _, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (k for (k,v), _, _, _, _, _ in data)))))
    k_size = max(map(len, chain.from_iterable(k for (k,v), _, _, _, _, _ in data)))
    #v_size = max(map(len, chain.from_iterable(v for (k,v), _, _, _, _, _ in data)))
    v_size = None
    query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    output_size = len(output_idx)
    #sentence_size = max(query_size, k_size)  # for the position
    if debug is True:
        log.info("Longest key length: {}".format(k_size))
        log.info("Longest value length: {}".format(v_size))
        log.info("Longest story length: {}".format(max_story_size))
        log.info("Average story length: {}".format(mean_story_size))
        log.info("Memory size: {}".format(memory_size))
    return memory_size, k_size, v_size, vocab_size, word_idx, output_size, output_idx


def vectorize_task_data(batch_size, data, debug, memory_size, random_state, sentence_size, test,
                        test_size, word_idx, log):
    S, Q, Y = vectorize_data(data, word_idx, sentence_size, memory_size)

    if debug:
        log.info("S : {}".format(S))
        log.info("Q : {}".format(Q))
        log.info("Y : {}".format(Y))
    trainS, valS, trainQ, valQ, trainY, valY = model_selection.train_test_split(S, Q, Y, test_size=test_size,
                                                                                random_state=random_state)
    testS, testQ, testY = vectorize_data(test, word_idx, sentence_size, memory_size)

    if debug:
        log.info("{}\n{}\n{}".format(S[0].shape, Q[0].shape, Y[0].shape))
        log.info("Training set shape {}".format(trainS.shape))

    # params
    n_train = trainS.shape[0]
    n_val = valS.shape[0]
    n_test = testS.shape[0]
    if debug:
        log.info("Training Size: {}".format(n_train))
        log.info("Validation Size: {}".format(n_val))
        log.info("Testing Size: {}".format(n_test))
    train_labels = np.argmax(trainY, axis=1)
    test_labels = np.argmax(testY, axis=1)
    val_labels = np.argmax(valY, axis=1)
    n_train_labels = train_labels.shape[0]
    n_val_labels = val_labels.shape[0]
    n_test_labels = test_labels.shape[0]

    if debug:
        log.info("Training Labels Size: {}".format(n_train_labels))
        log.info("Validation Labels Size: {}".format(n_val_labels))
        log.info("Testing Labels Size: {}".format(n_test_labels))

    train_batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
    val_batches = zip(range(0, n_val - batch_size, batch_size), range(batch_size, n_val, batch_size))
    test_batches = zip(range(0, n_test - batch_size, batch_size), range(batch_size, n_test, batch_size))

    return [trainS, trainQ, trainY], list(train_batches), [valS, valQ, valY], list(val_batches), [testS, testQ, testY], \
           list(test_batches)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    '''
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    '''
    S = []
    Q = []
    A = []
    VM = None  # vocabulary mask
    PL = None  # passage lengths
    SL = None  # sentences lengths
    QL = None  # query lengths

    for story, query, answer in data:
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        if len(ss) > memory_size:

            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ss, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ss) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                ss.remove(sent)
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A), VM, PL, SL, QL


def vectorize_data_clicr(data, word_idx, output_size, output_idx, sentence_size, memory_size):
    '''
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    vocab_mask marks which elements (=words/entities) in V are found in the particular document

    We also keep track of lengths of passages, sentences and queries, so that we can mask during
    vectorization the padded parts and ignore them in computation

    '''
    S = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    SM = []  # sentences mask
    QM = []  # query mask
    inv_w_idx = {v: k for k, v in word_idx.items()}

    for story, query, answer, _, _, _ in data:
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ss = []
        #ss_len = []
        for sentence in story:
            ls = max(0, sentence_size - len(sentence))
            sent = [word_idx[w] for w in sentence] + [0] * ls
            #sent_m = [1.] * len(sentence) + [0.] * ls
            if len(sent) > sentence_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:sentence_size]
                #sent_m = sent_m[:sentence_size]
            ss.append(sent)
            #ss_len.append(sent_m)

        if len(ss) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ss, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ss) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                #del_id = ss.index(sent)
                ss.remove(sent)
                #del ss_len[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            p_m = [1.] * len(ss) + [0.] * lm
            for _ in range(lm):
                ss.append([0] * sentence_size)
                #ss_len.append([0.] * sentence_size)
        y = np.zeros(output_size)
        #y = np.zeros(len(word_idx)+1)
        for a in answer:
            y[output_idx[a]] = 1
            #y[word_idx[a]] = 1

        vm = np.zeros_like(y)
        # mask for all words in vocab not part of the entities in the passage:
        # TODO this doesn't work for the no-ent setting
        ss_voc = {output_idx[inv_w_idx[i]] for i in set(np.array(ss).flatten()) if i!=0 and inv_w_idx[i] in output_idx}
        #ss_voc = {word_idx[inv_w_idx[i]] for i in set(np.array(ss).flatten()) if i != 0 and inv_w_idx[i] in word_idx}
        vm[list(ss_voc)] = 1.

        S.append(ss)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        SM = np.clip(np.array(S), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(S), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(SM), np.array(QM)

def vectorize_data_clicr_kv(data, word_idx, output_size, output_idx, k_size, memory_size):
    '''
    Vectorize stories (into keys and values) and queries.

    If a key/value length < key/value_size, it will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    vocab_mask marks which elements (=words/entities) in V are found in the particular document

    We also keep track of lengths of passages (=keys), keys, values and queries, so that we can mask during
    vectorization the padded parts and ignore them in computation

    '''
    K = []
    V = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    KM = []  # keys mask
    QM = []  # query mask
    inv_w_idx = {v: k for k, v in word_idx.items()}

    for (k,v), query, answer, _, _, _ in data:
        lq = max(0, k_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ks = []
        for win in k:
            ls = max(0, k_size - len(win))
            sent = [word_idx[w] for w in win] + [0] * ls
            if len(sent) > k_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:k_size]
            ks.append(sent)

        vs = [word_idx[val] for val in v]

        assert len(ks) == len(vs)
        if len(ks) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ks, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ks) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                del_id = ks.index(sent)
                del ks[del_id]
                del vs[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ks))
            p_m = [1.] * len(ks) + [0.] * lm
            for _ in range(lm):
                ks.append([0] * k_size)
                vs.append(0)
        y = np.zeros(output_size)
        for a in answer:
            y[output_idx[a]] = 1
            #y[word_idx[a]] = 1

        vm = np.zeros_like(y)
        # mask for all words in vocab not in the set of entities present in the values:
        # TODO this doesn't work for the no-ent setting
        vs_voc = {output_idx[inv_w_idx[i]] for i in set(np.array(vs).flatten()) if i!=0 and inv_w_idx[i] in output_idx}
        vm[list(vs_voc)] = 1.

        K.append(ks)
        V.append(vs)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        KM = np.clip(np.array(K), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(K), np.array(V), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(KM), np.array(QM)


def vectorize_data_clicr_kvatt(data, word_idx, output_size, output_idx, k_size, memory_size):
    '''
    Vectorize stories (into keys and values) and queries. Unlike in vectorization for KVMemNNs, we
    use here the output_idx for encoding the values, since we need values in prediction and for comparison
    to gold answers (which also use output_idx).

    If a key/value length < key/value_size, it will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    vocab_mask marks which elements (=words/entities) in V are found in the particular document

    We also keep track of lengths of passages (=keys), keys, values and queries, so that we can mask during
    vectorization the padded parts and ignore them in computation

    '''
    K = []
    V = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    KM = []  # keys mask
    QM = []  # query mask
    inv_w_idx = {v: k for k, v in word_idx.items()}

    for (k,v), query, answer, _, _, _ in data:
        lq = max(0, k_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ks = []
        for win in k:
            ls = max(0, k_size - len(win))
            sent = [word_idx[w] for w in win] + [0] * ls
            if len(sent) > k_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:k_size]
            ks.append(sent)

        vs = [output_idx[val] for val in v]

        assert len(ks) == len(vs)
        if len(ks) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ks, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ks) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                del_id = ks.index(sent)
                del ks[del_id]
                del vs[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ks))
            p_m = [1.] * len(ks) + [0.] * lm
            for _ in range(lm):
                ks.append([0] * k_size)
                vs.append(0)
        y = np.zeros(output_size)
        for a in answer:
            y[output_idx[a]] = 1
            #y[word_idx[a]] = 1

        vm = np.zeros_like(y)
        # mask for all words in vocab not in the set of entities present in the values:
        vs_voc = set(np.array(vs).flatten())
        vm[list(vs_voc)] = 1.

        K.append(ks)
        V.append(vs)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        KM = np.clip(np.array(K), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(K), np.array(V), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(KM), np.array(QM)


def vectorize_data_kvatt(data, word_idx, output_size, output_idx, k_size, memory_size):
    '''
    Vectorize stories (into keys and values) and queries. Unlike in vectorization for KVMemNNs, we
    use here the output_idx for encoding the values, since we need values in prediction and for comparison
    to gold answers (which also use output_idx).

    If a key/value length < key/value_size, it will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    vocab_mask marks which elements (=words/entities) in V are found in the particular document

    We also keep track of lengths of passages (=keys), keys, values and queries, so that we can mask during
    vectorization the padded parts and ignore them in computation

    '''
    K = []
    V = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    KM = []  # keys mask
    QM = []  # query mask
    inv_w_idx = {v: k for k, v in word_idx.items()}

    for (k,v), query, answer in data:
        lq = max(0, k_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ks = []
        for win in k:
            ls = max(0, k_size - len(win))
            sent = [word_idx[w] for w in win] + [0] * ls
            if len(sent) > k_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:k_size]
            ks.append(sent)

        vs = [output_idx[val] for val in v]

        assert len(ks) == len(vs)
        if len(ks) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ks, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ks) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                del_id = ks.index(sent)
                del ks[del_id]
                del vs[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ks))
            p_m = [1.] * len(ks) + [0.] * lm
            for _ in range(lm):
                ks.append([0] * k_size)
                vs.append(0)
        y = np.zeros(output_size)
        for a in answer:
            y[output_idx[a]] = 1
            #y[word_idx[a]] = 1

        vm = np.zeros_like(y)
        # mask for all words in vocab not in the set of entities present in the values:
        vs_voc = set(np.array(vs).flatten())
        vm[list(vs_voc)] = 1.

        K.append(ks)
        V.append(vs)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        KM = np.clip(np.array(K), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(K), np.array(V), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(KM), np.array(QM)

def vectorize_data_cbt_win(data, word_idx, output_size, win_size, memory_size):
    '''
    '''
    W = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    WM = []  # window mask
    QM = []  # query mask

    for wins, query, answer, _, _, _ in data:
        lq = max(0, win_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ws = []
        for win in wins:
            ls = max(0, win_size - len(win))
            sent = [word_idx[w] for w in win] + [0] * ls  # TODO pad zeros where truly missing, not only at end
            if len(sent) > win_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:win_size]
            ws.append(sent)

        if len(ws) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ws, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ws) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                del_id = ws.index(sent)
                del ws[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ws))
            p_m = [1.] * len(ws) + [0.] * lm
            for _ in range(lm):
                ws.append([0] * win_size)
        y = np.zeros(output_size)
        for a in answer:
            y[word_idx[a]] = 1

        vm = np.zeros_like(y)

        # vocab mask using only words in the passage
        vs_voc = set(np.array(ws).flatten())
        vm[list(vs_voc)] = 1.

        W.append(ws)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        WM = np.clip(np.array(W), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(W), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(WM), np.array(QM)


def vectorize_data_clicr_win(data, word_idx, output_size, win_size, memory_size, top_k_cand=10):
    '''
    '''
    if top_k_cand is not None:
        print("Topping n of cands to 10!")
    W = []
    Q = []
    A = []
    VM = []  # vocabulary mask
    PM = []  # passage mask
    WM = []  # window mask
    QM = []  # query mask
    inv_w_idx = {v: k for k, v in word_idx.items()}

    for wins, query, answer, _, _, _ in data:
        lq = max(0, win_size - len(query))
        q = [word_idx.get(w, word_idx["_UNK_"]) for w in query] + [0] * lq

        ws = []
        for win in wins:
            ls = max(0, win_size - len(win))
            sent = [word_idx.get(w, word_idx["_UNK_"]) for w in win] + [0] * ls  # TODO pad zeros where truly missing, not only at end
            if len(sent) > win_size:  # can happen in test/val as sentence_size is calculated on train
                sent = sent[:win_size]
            ws.append(sent)

        if len(ws) > memory_size:
            # TODO this is currently problematic as it relies on simple word match
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ws, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ws) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                del_id = ws.index(sent)
                del ws[del_id]
            p_m = [1.] * memory_size
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ws))
            p_m = [1.] * len(ws) + [0.] * lm
            for _ in range(lm):
                ws.append([0] * win_size)
        y = np.zeros(output_size)
        for a in answer:
            y[word_idx.get(a, word_idx["_UNK_"])] = 1

        vm = np.zeros_like(y)

        #vs_voc = set(np.array(ws).flatten())
        #TODO allow _UNK_ as the answer?--
        vs_voc = {i for i in set(np.array(ws).flatten()) if i!=0 and (inv_w_idx[i].startswith("@entity") or inv_w_idx[i] == "_UNK_")}

        if top_k_cand is not None:
            eff_top_k = top_k_cand
            a = word_idx.get(a, word_idx["_UNK_"])
            if a in vs_voc:
                vs_voc.remove(a)
                eff_top_k = eff_top_k - 1
            if len(vs_voc) > eff_top_k:
                vs_voc = np.random.choice(list(vs_voc), eff_top_k, replace=False)
                vs_voc = np.concatenate((vs_voc, np.array([a])))

        vm[list(vs_voc)] = 1.

        W.append(ws)
        Q.append(q)
        A.append(y)
        VM.append(vm)
        PM.append(p_m)
        WM = np.clip(np.array(W), 0., 1.)
        QM.append(np.clip(np.array(q), 0., 1.))

    return np.array(W), np.array(Q), np.array(A), np.array(VM), np.array(PM), np.array(WM), np.array(QM)



def generate_batches(batches_tr, batches_v, batches_te, train, val, test):
    train_batches = get_batch_from_batch_list(batches_tr, train)
    val_batches = get_batch_from_batch_list(batches_v, val)
    test_batches = get_batch_from_batch_list(batches_te, test)

    return train_batches, val_batches, test_batches


def get_batch_from_batch_list(batches_tr, train):
    trainS, trainQ, trainA = train
    trainA, trainQ, trainS = extract_tensors(trainA, trainQ, trainS)
    train_batches = []
    train_batches = construct_s_q_a_batch(batches_tr, train_batches, trainS, trainQ, trainA)
    return train_batches


def vectorized_batches(batches, data, word_idx, sentence_size, memory_size, output_size, output_idx, vectorizer=vectorize_data, shuffle=False):
    # batches are of form : [(0,2), (2,4),...]
    if shuffle:
        np.random.shuffle(batches)
    for s_batch, e_batch in batches:
        if vectorizer==vectorize_data:
            dataS, dataQ, dataA, dataVM, dataPM, dataSM, dataQM = vectorizer(data[s_batch:e_batch], word_idx, sentence_size, memory_size)
        else:
            dataS, dataQ, dataA, dataVM, dataPM, dataSM, dataQM = vectorizer(data[s_batch:e_batch], word_idx, output_size, output_idx, sentence_size, memory_size)

        dataA, dataQ, dataS, dataVM, dataPM, dataSM, dataQM = extract_tensors(dataA, dataQ, dataS, dataVM, dataPM, dataSM, dataQM)

        yield [list(dataS),
               list(dataQ),
               list(dataA),
               list(dataVM) if dataVM is not None else None,
               list(dataPM) if dataPM is not None else None,
               list(dataSM) if dataSM is not None else None,
               list(dataQM) if dataQM is not None else None
               ]

def vectorized_batches_kv(batches, data, word_idx, k_size, memory_size, output_size, output_idx, vectorizer=vectorize_data_clicr_kv, shuffle=False):
    # batches are of form : [(0,2), (2,4),...]
    if shuffle:
        np.random.shuffle(batches)
    for s_batch, e_batch in batches:
        dataK, dataV, dataQ, dataA, dataVM, dataPM, dataKM, dataQM = vectorizer(data[s_batch:e_batch], word_idx, output_size, output_idx, k_size, memory_size)
        dataA, dataQ, dataK, dataV, dataVM, dataPM, dataKM, dataQM = extract_tensors_kv(dataA, dataQ, dataK, dataV, dataVM, dataPM, dataKM, dataQM)

        yield [list(dataK),
               list(dataV),
               list(dataQ),
               list(dataA),
               list(dataVM) if dataVM is not None else None,
               list(dataPM) if dataPM is not None else None,
               list(dataKM) if dataKM is not None else None,
               list(dataQM) if dataQM is not None else None
               ]


def vectorized_batches_win(batches, data, word_idx, win_size, memory_size, output_size, vectorizer=vectorize_data_cbt_win, shuffle=False):
    # batches are of form : [(0,2), (2,4),...]
    if shuffle:
        np.random.shuffle(batches)
    for s_batch, e_batch in batches:
        dataW, dataQ, dataA, dataVM, dataPM, dataWM, dataQM = vectorizer(data[s_batch:e_batch], word_idx, output_size, win_size, memory_size)
        dataA, dataQ, dataW, dataVM, dataPM, dataWM, dataQM = extract_tensors_win(dataA, dataQ, dataW, dataVM, dataPM, dataWM, dataQM)

        yield [list(dataW),
               list(dataQ),
               list(dataA),
               list(dataVM) if dataVM is not None else None,
               list(dataPM) if dataPM is not None else None,
               list(dataWM) if dataWM is not None else None,
               list(dataQM) if dataQM is not None else None
               ]


def extract_tensors(A, Q, S, VM, PM, SM, QM):
    A = torch.from_numpy(A).type(long_tensor_type)
    S = torch.from_numpy(S).type(float_tensor_type)
    Q = np.expand_dims(Q, 1)
    Q = torch.from_numpy(Q).type(long_tensor_type)
    VM = torch.from_numpy(VM).type(float_tensor_type) if VM is not None else None
    PM = torch.from_numpy(PM).type(float_tensor_type) if PM is not None else None
    SM = torch.from_numpy(SM).type(float_tensor_type) if SM is not None else None
    QM = torch.from_numpy(QM).type(float_tensor_type) if QM is not None else None
    return A, Q, S, VM, PM, SM, QM


def extract_tensors_kv(A, Q, K, V, VM, PM, KM, QM):
    A = torch.from_numpy(A).type(long_tensor_type)
    K = torch.from_numpy(K).type(long_tensor_type)
    V = torch.from_numpy(V).type(long_tensor_type)
    Q = np.expand_dims(Q, 1)
    Q = torch.from_numpy(Q).type(long_tensor_type)
    VM = torch.from_numpy(VM).type(float_tensor_type) if VM is not None else None
    PM = torch.from_numpy(PM).type(float_tensor_type) if PM is not None else None
    KM = torch.from_numpy(KM).type(float_tensor_type) if KM is not None else None
    QM = torch.from_numpy(QM).type(float_tensor_type) if QM is not None else None
    return A, Q, K, V, VM, PM, KM, QM


def extract_tensors_win(A, Q, W, VM, PM, WM, QM):
    A = torch.from_numpy(A).type(long_tensor_type)
    W = torch.from_numpy(W).type(long_tensor_type)
    Q = np.expand_dims(Q, 1)
    Q = torch.from_numpy(Q).type(long_tensor_type)
    VM = torch.from_numpy(VM).type(float_tensor_type) if VM is not None else None
    PM = torch.from_numpy(PM).type(float_tensor_type) if PM is not None else None
    WM = torch.from_numpy(WM).type(float_tensor_type) if WM is not None else None
    QM = torch.from_numpy(QM).type(float_tensor_type) if QM is not None else None
    return A, Q, W, VM, PM, WM, QM


def construct_s_q_a_batch(batches, batched_objects, S, Q, A):
    for batch in batches:
        # batches are of form : [(0,2), (2,4),...]
        answer_batch = []
        story_batch = []
        query_batch = []
        for j in range(batch[0], batch[1]):
            answer_batch.append(A[j])
            story_batch.append(S[j])
            query_batch.append(Q[j])
        batched_objects.append([story_batch, query_batch, answer_batch])

    return batched_objects


def process_eval_data(data_dir, task_num, word_idx, sentence_size, vocab_size, log, memory_size=50, batch_size=2,
                      test_size=.1, debug=True, joint_training=0):
    random_state = None
    data, test, vocab = load_data(data_dir, joint_training, task_num)

    if (joint_training == 0):
        memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values(data=data, debug=debug,
                                                                                      memory_size=memory_size,
                                                                                      vocab=vocab)
    train_set, train_batches, val_set, val_batches, test_set, test_batches = \
        vectorize_task_data(batch_size, data, debug, memory_size, random_state,
                            sentence_size, test, test_size, word_idx, log)

    return train_batches, val_batches, test_batches, train_set, val_set, test_set, \
           sentence_size, vocab_size, memory_size, word_idx


def get_position_encoding(batch_size, sentence_size, embedding_size):
    '''
    Position Encoding 
    '''
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    enc_vec = torch.from_numpy(np.transpose(encoding)).type(float_tensor_type)
    lis = []
    for _ in range(batch_size):
        lis.append(enc_vec)
    enc_vec = Variable(torch.stack(lis))
    return enc_vec


def weight_update(name, param):
    update = param.grad
    weight = param.data
    print(name, (torch.norm(update) / torch.norm(weight)).data[0])


def load_w2v(fn):
    emb_idx = {}
    with open(fn) as fh:
        m, n = map(eval, fh.readline().strip().split())
        e_m = np.random.normal(size=(m, n), loc=0, scale=0.1)
        for c, l in enumerate(fh):
            w, *e = l.strip().split()
            if len(e) != n:
                print("Incorrect embedding dimension, skipping.")
                continue
            if not w or not e:
                print("Empty w or e.")
            emb_idx[w] = c
            e_m[c] = e
    return e_m, emb_idx, n


def update_vectors(pretr_embs, pretr_emb_idx, embs, word_idx):
    c = 0
    for w, i in word_idx.items():
        if w.startswith("@entity"):
            w_l = deentitize(w).split(" ")
            w_idx = [pretr_emb_idx[w] for w in w_l if w in pretr_emb_idx]
            if not w_idx:
                continue
            embs[i] = np.average(pretr_embs[w_idx], axis=0)
            c+=1
        else:
            if w not in pretr_emb_idx:
                continue
            embs[i] = pretr_embs[pretr_emb_idx[w]]
    print("Updated {} entity vectors".format(c))
    return embs


def load_emb(fn, word_idx, freeze=False, ent_setup="ent"):
    pretr_embs, pretr_emb_idx, n = load_w2v(fn)
    # build rep. for entities by averaging word vectors
    embs = np.random.normal(size=(len(word_idx)+1, n), loc=0, scale=0.1)
    embs = update_vectors(pretr_embs, pretr_emb_idx, embs, word_idx)
    embs_tensor = nn.Embedding.from_pretrained(float_tensor_type(embs), freeze=freeze)

    return embs_tensor, n


def evaluate_clicr(test_file, preds_file, extended=False,
                   emb_file="/nas/corpora/accumulate/clicr/embeddings/b2257916-6a9f-11e7-aa74-901b0e5592c8/embeddings",
                   downcase=True):
    results = subprocess.check_output(
        "python3 ~/Apps/bmj_case_reports/evaluate.py -test_file {test_file} -prediction_file {preds_file} -embeddings_file {emb_file} {downcase} {extended}".format(
            test_file=test_file, preds_file=preds_file, emb_file=emb_file, downcase="-downcase" if downcase else "",
            extended="-extended" if extended else ""), shell=True)
    return results


def deentitize(s):
    """
    :param s: e.g. "@entityparalysis_of_the_lower_limbs"
    :return: "paralysis of the lower limbs"
    """
    e_m = "@entity"
    assert s.startswith("@entity")

    return s[len(e_m):].replace("_", " ")
