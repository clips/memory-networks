import argparse
import os
import re
import string
from collections import Counter

from tqdm import tqdm
import numpy as np

from util import save_json, read_cbt, load_cbt_win


def down(_, downcase=False):
    return _.lower() if downcase else _


def line_reader(f, skip=0):
    with open(f) as in_f:
        for c, l in enumerate(in_f, 1):
            if c <= skip:
                continue
            yield l


class VocabBuild():
    def __init__(self, filename, sep=" ", downcase=False):
        self.filename = filename
        self.sep = sep
        self.downcase = downcase  # whether the embs have been lowercased

        self.w_index = {}
        self.inv_w_index = {}
        self.W = None

    def read(self):
        """
        Reads word2vec-format embeddings.
        """
        ws = []
        with open(self.filename) as in_f:
            m, n = map(eval, in_f.readline().strip().split())
        e_m = np.zeros((m, n))
        for c, l in enumerate(line_reader(self.filename, skip=1)):  # skip dimensions
            w, *e = l.strip().split()
            # assert len(e) == n
            if len(e) != n:
                print("Incorrect embedding dimension, skipping.")
                continue
            if not w or not e:
                print("Empty w or e.")
            ws.append(w)
            e_m[c] = e
        # assert len(ws) == e_m.shape[0]
        self.w_index = {w: c for c, w in enumerate(ws)}
        self.inv_w_index = {v: k for k, v in self.w_index.items()}
        self.W = e_m

    def lookup(self, w, output_nan=False):
        if down(w, self.downcase) in self.w_index:
            idx = self.w_index[down(w, self.downcase)]
        else:
            if output_nan:
                idx = 0
            else:
                idx = None

        return idx

    def line_to_seq(self, toks, output_nan=False):
        seq = []
        for w in toks:
            idx = self.lookup(w, output_nan=output_nan)
            if idx is None:
                continue
            seq.append(idx)

        return seq

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_json(self.w_index, "{}/w_index.json".format(save_dir))
        save_json(self.inv_w_index, "{}/inv_w_index.json".format(save_dir))


def normalize_answer(s, lemmatizer_comm=None):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        if type(text) == list:
            print()
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, comm=None):
    prediction_tokens = normalize_answer(prediction, comm).split()
    ground_truth_tokens = normalize_answer(ground_truth, comm).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, comm=None):
    return normalize_answer(prediction, comm) == normalize_answer(ground_truth, comm)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, comm=None):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, comm)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    insts = read_cbt(dataset)
    f1 = exact_match = total = 0
    print("evaluating")
    for c, inst in tqdm(enumerate(insts)):
        total += 1
        ground_truths = [inst[2]]  # here, a single answer
        prediction = predictions[c]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    assert exact_match <= f1
    scores = {'exact_match': exact_match, 'f1': f1}

    return scores


def vectorize_contexts_of_words(sents, v, cands, win_size=3):
    targets = []
    T = []  # n_words*(2*win_size)

    for sent, w in sents:
        contexts = v.line_to_seq(sent, output_nan=True)
        if len(contexts) < 2:
            continue
        for _ in range(2 * win_size - len(contexts)):  # padding for start/end sent, exclude cand
            contexts.append(0)  # special out of seq idx
            targets.append(w)
            T.append(contexts)
    if T:
        T_w_summed = v.W[np.array(T)].sum(axis=1)  # n_words*d
        assert len(targets) == T_w_summed.shape[0]
    else:
        T_w_summed = v.W[np.array([[0]*win_size])].sum(axis=1)  # n_words*d
        targets.append(cands[0][0])
        print("empty")


    return targets, T_w_summed


def vectorize_query(q, v, win_size=3):
    contexts = v.line_to_seq(q[0])
    for _ in range(2 * win_size - len(contexts)):  # padding for start/end sent, exclude cand
        contexts.append(0)  # special out of seq idx
    assert len(contexts) == 2 * win_size
    q_w_summed = v.W[np.array(contexts)].sum(axis=0)  # d*1

    return q_w_summed


def distance_baseline(dataset, embeddings_file, downcase, context_vectorize_fun, win_size):
    v = VocabBuild(embeddings_file, downcase=downcase)
    v.read()
    #insts = read_cbt(dataset)
    insts, _ = load_cbt_win(dataset, win_size=win_size, include_cand=False)
    predictions = []
    print("obtaining predictions")
    for c, inst in enumerate(tqdm(insts)):
        wins, q_win, a, cands, cloze, id = inst
        targets, C = context_vectorize_fun(wins, v, cands, win_size=win_size)  # n_words*d
        query_repr = vectorize_query(q_win, v, win_size=win_size)
        idx = best_answer(C, query_repr)
        predictions.append(targets[idx])

    return predictions


def best_answer(context_matrix, query_vector):
    return cosines(context_matrix, query_vector).argmax()


def cosines(W, W2):
    if W2.ndim == 2:
        scores = []
        for w_emb in W2:
            scores.append(cosines(W, w_emb))
        return np.array(scores)
    w_emb_norm = np.linalg.norm(W2)
    return np.dot(W, W2) / (np.linalg.norm(W, axis=1) * w_emb_norm)


def print_scores(scores):
    """
    :param scores: {"method1": score, ...}
    """
    print("{}\t{:.1f}".format("exact_match", scores["exact_match"]))
    print("{}\t{:.1f}".format("f1", scores["f1"]))

    for method, score in sorted(scores.items()):
        if method == "exact_match" or method == "f1":
            continue
        else:
            print("{}\t{:.3f}".format(method, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply some simple baselines.')
    parser.add_argument('-test_file',
                        default='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/CBTest/data/cbtest_P_test_2500ex.txt')
    parser.add_argument('-embeddings_file', help='Embeddings in w2v txt format.',
                        default='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/news_embs/embs/1/embeddings')
    parser.add_argument('-downcase',
                        help="Only for distance baselines. Should be set to true if the embedding vocabulary is lowercased.",
                        action="store_true")
    parser.add_argument("-win_size", help="Window size to each side for the embedding baselines.", default=5, type=int)
    args = parser.parse_args()

    print(args.test_file)
    print(args.embeddings_file)

    print("Obtaining baseline predictions...")
    predictions_distance_words = distance_baseline(args.test_file, args.embeddings_file, args.downcase,
                                                   vectorize_contexts_of_words, win_size=args.win_size)
    print("sim-entity OK")

    scores_distance_words = evaluate(args.test_file, predictions_distance_words)

    print("\nsim-entity:")
    print_scores(scores_distance_words)
