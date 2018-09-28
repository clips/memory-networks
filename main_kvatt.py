import argparse
from datetime import datetime
import os

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np

from logger import get_logger
from net import N2N, KVN2N, KVAtt
from util import long_tensor_type, vectorize_data_clicr, vectorized_batches, vectorize_data, evaluate_clicr, save_json, \
    get_q_ids_clicr, remove_missing_preds, deentitize, process_data_clicr_kv, vectorized_batches_kv, \
    vectorize_data_clicr_kv, vectorize_data_clicr_kvatt
from util import process_data, process_data_clicr


def train_network_kvatt(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx, sentence_size,
                  vocab_size, story_size, output_size, output_idx, save_model_path, args, log, attention_sum):

    net = KVAtt(args.batch_size, args.embed_size, vocab_size, story_size=story_size, args=args,
                  word_idx=word_idx, output_size=output_size)
    positional = False  # don't use positional encoding for KV network
    if torch.cuda.is_available() and args.cuda == 1:
        net = net.cuda()
    criterion = torch.nn.NLLLoss()
    log.info("{}\n".format(net))
    if not args.freeze_pretrained_word_embed:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer.zero_grad()
    vectorizer = vectorize_data_clicr_kvatt
    running_loss = 0.0
    best_val_acc_yet = 0.0
    for current_epoch in range(args.epochs):
        k_size = sentence_size
        train_batch_gen = vectorized_batches_kv(train_batches_id, data, word_idx, k_size, story_size,
                                                 output_size, output_idx, vectorizer, shuffle=args.shuffle)
        current_len = 0
        current_correct = 0
        for batch, (s_batch, _) in zip(train_batch_gen, train_batches_id):
            idx_out, idx_true, out, att_probs = epoch_kvatt(batch, net, args.inspect, positional, attention_sum)
            current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
            if not args.freeze_pretrained_word_embed:
                loss = criterion(out, idx_true)
                loss.backward()
                clip_grad_norm_(net.parameters(), 40)
                running_loss += loss
                optimizer.step()
                optimizer.zero_grad()
        if not args.freeze_pretrained_word_embed:
            if current_epoch % args.log_epochs == 0:
                accuracy = 100 * (current_correct / current_len)
                if args.mode == "kv":
                    val_acc, val_cor, val_tot = calculate_loss_and_accuracy_kvatt(net, val_batches_id, val_data, word_idx, sentence_size, story_size,
                                                                        output_size, output_idx, vectorizer, args.inspect, positional, attention_sum)
                log.info("Epochs: {}, Train Accuracy: {:.3f}, Loss: {:.3f}, Val_Acc:{:.3f} ({}/{})".format(current_epoch, accuracy,
                                                                                    running_loss.item(),
                                                                                    val_acc, val_cor, val_tot))
                if best_val_acc_yet <= val_acc and args.save_model:
                    torch.save(net.state_dict(), save_model_path)
                    best_val_acc_yet = val_acc

            if current_epoch % args.anneal_epoch == 0 and current_epoch != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / args.anneal_factor
            running_loss = 0.0


def epoch_kvatt(batch, net, inspect=False, positional=True, attention_sum=False):
    key_batch = batch[0]
    value_batch = batch[1]
    query_batch = batch[2]
    answer_batch = batch[3]
    vocabmask_batch = batch[4]
    pasmask_batch = batch[5]
    keymask_batch = batch[6]
    querymask_batch = batch[7]

    A = Variable(torch.stack(answer_batch, dim=0), requires_grad=False).type(long_tensor_type)
    _, idx_true = torch.max(A, 1)
    idx_true = torch.squeeze(idx_true)

    K = torch.stack(key_batch, dim=0)
    V = torch.stack(value_batch, dim=0)
    Q = torch.stack(query_batch, dim=0)
    VM = torch.stack(vocabmask_batch, dim=0) if vocabmask_batch is not None else None
    PM = torch.stack(pasmask_batch, dim=0) if pasmask_batch is not None else None
    KM = torch.stack(keymask_batch, dim=0) if keymask_batch is not None else None
    QM = torch.stack(querymask_batch, dim=0) if querymask_batch is not None else None

    out, idx_out, att_probs = net(K, V, Q, VM, PM, KM, QM, inspect, positional=positional, attention_sum=attention_sum)

    return idx_out, idx_true, out, att_probs

def update_counts(current_correct, current_len, idx_out, idx_true):
    batch_len, correct = count_predictions(idx_true, idx_out)
    current_len += batch_len
    current_correct += correct
    return current_correct, current_len


def count_predictions(labels, predicted):
    batch_len = len(labels)
    correct = float((predicted == labels).sum())
    return batch_len, correct


def calculate_loss_and_accuracy_kvatt(net, batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer, inspect=False, positional=False, attention_sum=False):
    batch_gen = vectorized_batches_kv(batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer)
    current_len = 0
    current_correct = 0
    for batch in batch_gen:
        idx_out, idx_true, out, att_probs = epoch_kvatt(batch, net, inspect, positional, attention_sum)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    return 100 * (current_correct / current_len), current_correct, current_len


def eval_network(vocab_size, story_size, sentence_size, model, word_idx, output_size, output_idx, test_batches_id, test, log, logdir, args, cuda=0., test_q_ids=None, max_inspect=5, ignore_missing_preds=False, attention_sum=False):
    log.info("Evaluating")
    net = KVAtt(args.batch_size, args.embed_size, vocab_size, story_size=story_size, args=args,
                  word_idx=word_idx, output_size=output_size)
    positional = False  # don't use positional encoding for KV network
    if model is not None:
        net.load_state_dict(torch.load(model))
    inv_output_idx = {v: k for k, v in output_idx.items()}
    if torch.cuda.is_available() and cuda == 1:
        net = net.cuda()
    vectorizer = vectorize_data_clicr_kvatt
    k_size = sentence_size
    test_batch_gen = vectorized_batches_kv(test_batches_id, test, word_idx, k_size, story_size,
                                                output_size, output_idx, vectorizer, shuffle=args.shuffle)
    current_len = 0
    current_correct = 0
    preds = {} if args.dataset == "clicr" else None

    for batch, (s_batch, _) in zip(test_batch_gen, test_batches_id):
        idx_out, idx_true, out, att_probs = epoch_kvatt(batch, net, args.inspect, positional, attention_sum)
        if preds is not None:
            for c, i in enumerate(idx_out):
                # {query_id: answer}
                preds[test[s_batch+c][5]] = deentitize(inv_output_idx[i.item()])
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    # clicr detailed evaluation
    if args.dataset=="clicr":
        missing = test_q_ids - preds.keys()
        log.info("\n{} predictions missing out of {}.".format(len(missing), len(test_q_ids)))
        if ignore_missing_preds:
            log.info("Ignoring missing predictions.")
            new_test = remove_missing_preds(args.data_dir + "test1.0.json", preds.keys())
            test_file = logdir + "/reduced_test.json"
            save_json(new_test, test_file)
        else:
            for q_id in missing:
                preds[q_id] = ""
            test_file = args.data_dir + "test1.0.json"
        preds_file = logdir + "/preds.json"
        save_json(preds, preds_file)
        results = evaluate_clicr(test_file, preds_file, extended=True, downcase=True)
        log.info(results.decode())

    accuracy = 100 * (current_correct / current_len)
    log.info("Accuracy : {}".format(accuracy))


def model_path(dir, args):
    if args.joint_training == 1:
        saved_model_filename = "joint_model.model"
    elif args.dataset == "babi":
        saved_model_filename = str(args.task_number) + "_model.model"
    else:
        saved_model_filename = "model.model"
    saved_model_path = os.path.join(dir, saved_model_filename)
    return saved_model_path


def main():
    arg_parser = argparse.ArgumentParser(description="parser for End-to-End Memory Networks")

    arg_parser.add_argument("--anneal-epoch", type=int, default=25,
                            help="anneal every [anneal-epoch] epoch, default: 25")
    arg_parser.add_argument("--anneal-factor", type=int, default=2,
                            help="factor to anneal by every 'anneal-epoch(s)', default: 2")
    arg_parser.add_argument("--attention-sum", action="store_true", help="Flag to sum attention probs for the same entity.")
    arg_parser.add_argument("--average-embs", type=int, default=1, help="Flag to average context embs instead of summing.")
    arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training, default: 32")
    arg_parser.add_argument("--cuda", type=int, default=0, help="train on GPU, default: 0")
    arg_parser.add_argument("--data-dir", type=str, default="./data/tasks_1-20_v1-2/en",
                            help="path to folder from where data is loaded")
    arg_parser.add_argument("--dataset", type=str, help="babi or clicr")
    arg_parser.add_argument("--debug", action="store_true", help="Flag for debugging purposes")
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimensions, default: 25")
    arg_parser.add_argument("--ent-setup", type=str, default="ent", help="How to treat entities in CliCR.")
    arg_parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default: 100")
    arg_parser.add_argument("--eval", type=int, default=1, help="evaluate after training, default: 1")
    arg_parser.add_argument("--freeze-pretrained-word-embed", action="store_true",
                            help="will prevent the pretrained word embeddings from being updated")
    arg_parser.add_argument("--hops", type=int, default=1, help="Number of hops to make: 1, 2 or 3; default: 1 ")
    arg_parser.add_argument("--ignore-missing-preds", action="store_true",
                            help="Whether to remove the missing predictions from the test during evaluation.")
    arg_parser.add_argument("--inspect", action="store_true", help="Flag to inspect attention and output distribution.")
    arg_parser.add_argument("--joint-training", type=int, default=0, help="joint training flag, default: 0")
    arg_parser.add_argument("--load-model-path", type=str, help="File path for the model.")
    arg_parser.add_argument("--log-epochs", type=int, default=4,
                            help="Number of epochs after which to log progress, default: 4")
    arg_parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default: 0.01")
    arg_parser.add_argument("--max-n-load", type=int, help="maximum number of clicr documents to use, for debugging")
    arg_parser.add_argument("--memory-size", type=int, default=50, help="upper limit on memory size, default: 50")
    arg_parser.add_argument("--mode", type=str, default="standard", help="standard | kv")
    arg_parser.add_argument("--pretrained-word-embed", type=str,
                            help="path to the txt file with word embeddings")  # "/nas/corpora/accumulate/clicr/embeddings/4bfb98c2-688e-11e7-aa74-901b0e5592c8/embeddings"
    arg_parser.add_argument("--save-model", action="store_true")
    arg_parser.add_argument("--shuffle", action="store_true")
    arg_parser.add_argument("--task-number", type=int, default=1, help="Babi task to process, default: 1")
    arg_parser.add_argument("--train", type=int, default=1)
    arg_parser.add_argument("--win-size-kv", type=int, default=3, help="Size of the key window for one side.")


    args = arg_parser.parse_args()
    if args.dataset == "clicr" and args.eval==1: # load all gold query ids in the test
        test_q_ids = get_q_ids_clicr(args.data_dir + "/test1.0.json")
    if args.eval == 1:
        args.save_model = True
    exp_dir = "./experiments/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if args.train == 0 and args.eval == 1 and args.load_model_path != "None":
        logdir = os.path.dirname(args.load_model_path)
        log = get_logger(logdir + "/log_eval")
    else:
        logdir = "{}{}".format(exp_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log = get_logger(logdir + "/log")

    for argk, argv in sorted(vars(args).items()):
        log.info("{}: {}".format(argk, argv))
    log.info("")
    print("Output to {}".format(logdir))
    save_model_path = model_path(logdir, args)

    # load data
    data, val_data, test_data, k_size, v_size, vocab_size, story_size, word_idx, output_size, output_idx = process_data_clicr_kv(args, log=log)
    if args.pretrained_word_embed:
        log.info("Using pretrained word embeddings: {}".format(args.pretrained_word_embed))
    else:
        log.info("Using random initializativectorized_batches_kvon.")
    # get batch indices
    # TODO: don't leave out instances
    n_train = len(data)
    n_val = len(val_data)
    n_test = len(test_data)
    train_batches_id = list(
        zip(range(0, n_train - args.batch_size, args.batch_size), range(args.batch_size, n_train, args.batch_size)))
    val_batches_id = list(
        zip(range(0, n_val - args.batch_size, args.batch_size), range(args.batch_size, n_val, args.batch_size)))
    test_batches_id = list(
        zip(range(0, n_test - args.batch_size, args.batch_size), range(args.batch_size, n_test, args.batch_size)))
    if args.train == 1:
        train_network_kvatt(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx,
                      k_size, story_size=story_size,
                      vocab_size=vocab_size, output_size=output_size, output_idx=output_idx, save_model_path=save_model_path, args=args, log=log, attention_sum=args.attention_sum)
    if args.eval == 1:
        if args.train == 1:
            model = save_model_path
        else:
            #model = args.load_model_path
            model = None
        eval_network(vocab_size, story_size, k_size, model, word_idx, output_size, output_idx, test_batches_id, test_data, log, logdir, args, cuda=args.cuda, test_q_ids=test_q_ids, ignore_missing_preds=args.ignore_missing_preds, attention_sum=args.attention_sum)


if __name__ == '__main__':
    main()
