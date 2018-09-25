import argparse
from datetime import datetime
import os

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from logger import get_logger
from net import N2N, KVN2N
from util import long_tensor_type, vectorize_data_clicr, vectorized_batches, vectorize_data, evaluate_clicr, save_json, \
    get_q_ids_clicr, remove_missing_preds, deentitize, process_data_clicr_kv, vectorized_batches_kv, \
    vectorize_data_clicr_kv
from util import process_data, process_data_clicr



def train_network(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx, sentence_size,
                  vocab_size, story_size, output_size, output_idx, save_model_path, args, log, max_inspect=15):
    if args.inspect:
        inv_output_idx = {v: k for k, v in output_idx.items()}
    if args.mode == "kv":
        net = KVN2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args,
                  word_idx=word_idx, output_size=output_size)
    else:
        net = N2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args, word_idx=word_idx, output_size=output_size)
    if torch.cuda.is_available() and args.cuda == 1:
        net = net.cuda()
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    #for name, param in net.named_parameters():
    #    if param.requires_grad:
    #        log.info("{}\t{}".format(name, param.data))
    log.info("{}\n".format(net))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()
    if args.dataset == "clicr":
        if args.mode == "standard":
            vectorizer = vectorize_data_clicr
        elif args.mode == "kv":
            vectorizer = vectorize_data_clicr_kv
    elif args.dataset == "babi":
        vectorizer = vectorize_data
    else:
        raise NotImplementedError

    running_loss = 0.0
    best_val_acc_yet = 0.0
    for current_epoch in range(args.epochs):
        if args.inspect:
            n_inspect = 0
        if args.mode == "standard":
            train_batch_gen = vectorized_batches(train_batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer, shuffle=args.shuffle)
        elif args.mode == "kv":
            k_size = sentence_size
            train_batch_gen = vectorized_batches_kv(train_batches_id, data, word_idx, k_size, story_size,
                                                 output_size, output_idx, vectorizer, shuffle=args.shuffle)
        current_len = 0
        current_correct = 0
        for batch, (s_batch, _) in zip(train_batch_gen, train_batches_id):
            if args.mode == "kv":
                idx_out, idx_true, out, att_probs = epoch_kv(batch, net, args.inspect)
            else:
                idx_out, idx_true, out, att_probs = epoch(batch, net, args.inspect)

            #if current_epoch == args.epochs - 1 and args.inspect and n_inspect < max_inspect:
            if args.inspect and n_inspect < max_inspect:
                if args.mode == "kv":
                    inspect_kv(out, idx_true, os.path.dirname(save_model_path), current_epoch, s_batch, att_probs,
                            inv_output_idx, data, args, log)
                else:
                    inspect(out, idx_true, os.path.dirname(save_model_path), current_epoch, s_batch, att_probs, inv_output_idx, data, args, log)
                n_inspect += 1
            loss = criterion(out, idx_true)
            loss.backward()
            clip_grad_norm_(net.parameters(), 40)
            running_loss += loss
            current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
            optimizer.step()
            optimizer.zero_grad()
        if current_epoch % args.log_epochs == 0:
            accuracy = 100 * (current_correct / current_len)
            if args.mode == "kv":
                val_acc, val_cor, val_tot = calculate_loss_and_accuracy_kv(net, val_batches_id, val_data, word_idx, sentence_size, story_size,
                                                                    output_size, output_idx, vectorizer, args.inspect)
            else:
                val_acc, val_cor, val_tot = calculate_loss_and_accuracy(net, val_batches_id, val_data, word_idx,
                                                                    sentence_size, story_size,
                                                                    output_size, output_idx, vectorizer, args.inspect)
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


def epoch(batch, net, inspect=False):
    story_batch = batch[0]
    query_batch = batch[1]
    answer_batch = batch[2]
    vocabmask_batch = batch[3]
    pasmask_batch = batch[4]
    sentmask_batch = batch[5]
    querymask_batch = batch[6]

    A = Variable(torch.stack(answer_batch, dim=0), requires_grad=False).type(long_tensor_type)
    _, idx_true = torch.max(A, 1)
    idx_true = torch.squeeze(idx_true)

    S = torch.stack(story_batch, dim=0)
    Q = torch.stack(query_batch, dim=0)
    VM = torch.stack(vocabmask_batch, dim=0) if vocabmask_batch is not None else None
    PM = torch.stack(pasmask_batch, dim=0) if pasmask_batch is not None else None
    SM = torch.stack(sentmask_batch, dim=0) if sentmask_batch is not None else None
    QM = torch.stack(querymask_batch, dim=0) if querymask_batch is not None else None

    if inspect:
        out, att_probs = net(S, Q, VM, PM, SM, QM, inspect)
    else:
        out = net(S, Q, VM, PM, SM, QM, inspect)

    _, idx_out = torch.max(out, 1)
    return idx_out, idx_true, out, att_probs if inspect else None


def epoch_kv(batch, net, inspect=False):
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

    if inspect:
        out, att_probs = net(K, V, Q, VM, PM, KM, QM, inspect, positional=False)
    else:
        out = net(K, V, Q, VM, PM, KM, QM, inspect, positional=False)

    _, idx_out = torch.max(out, 1)
    return idx_out, idx_true, out, att_probs if inspect else None


def update_counts(current_correct, current_len, idx_out, idx_true):
    batch_len, correct = count_predictions(idx_true, idx_out)
    current_len += batch_len
    current_correct += correct
    return current_correct, current_len


def count_predictions(labels, predicted):
    batch_len = len(labels)
    correct = float((predicted == labels).sum())
    return batch_len, correct


def calculate_loss_and_accuracy(net, batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer, inspect=False):
    batch_gen = vectorized_batches(batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer)
    current_len = 0
    current_correct = 0
    for batch in batch_gen:
        idx_out, idx_true, out, _ = epoch(batch, net, inspect)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    return 100 * (current_correct / current_len), current_correct, current_len


def calculate_loss_and_accuracy_kv(net, batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer, inspect=False):
    batch_gen = vectorized_batches_kv(batches_id, data, word_idx, sentence_size, story_size, output_size, output_idx, vectorizer)
    current_len = 0
    current_correct = 0
    for batch in batch_gen:
        idx_out, idx_true, out, _ = epoch_kv(batch, net, inspect)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    return 100 * (current_correct / current_len), current_correct, current_len


def eval_network(vocab_size, story_size, sentence_size, model, word_idx, output_size, output_idx, test_batches_id, test, log, logdir, args, cuda=0., test_q_ids=None, max_inspect=5, ignore_missing_preds=False):
    log.info("Evaluating")
    if args.mode == "kv":
        net = KVN2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args,
                  word_idx=word_idx, output_size=output_size)
    else:
        net = N2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args, word_idx=word_idx, output_size=output_size)
    net.load_state_dict(torch.load(model))
    inv_output_idx = {v: k for k, v in output_idx.items()}
    if args.inspect:
        n_inspect = 0

    if torch.cuda.is_available() and cuda == 1:
        net = net.cuda()
    if args.dataset == "clicr":
        if args.mode == "standard":
            vectorizer = vectorize_data_clicr
        elif args.mode == "kv":
            vectorizer = vectorize_data_clicr_kv
    elif args.dataset == "babi":
        vectorizer = vectorize_data
    else:
        raise NotImplementedError
    if args.mode == "standard":
        test_batch_gen = vectorized_batches(test_batches_id, data, word_idx, sentence_size, story_size, output_size,
                                             output_idx, vectorizer, shuffle=args.shuffle)
    elif args.mode == "kv":
        k_size = sentence_size
        test_batch_gen = vectorized_batches_kv(test_batches_id, data, word_idx, k_size, story_size,
                                                output_size, output_idx, vectorizer, shuffle=args.shuffle)
    current_len = 0
    current_correct = 0
    preds = {} if args.dataset == "clicr" else None

    for batch, (s_batch, _) in zip(test_batch_gen, test_batches_id):
        if args.mode == "kv":
            idx_out, idx_true, out, att_probs = epoch_kv(batch, net, args.inspect)
        else:
            idx_out, idx_true, out, att_probs = epoch(batch, net, args.inspect)
        if args.inspect and n_inspect < max_inspect:
            if args.mode == "kv":
                inspect_kv(out, idx_true, os.path.dirname(save_model_path), current_epoch, s_batch, att_probs,
                           inv_output_idx, data, args, log)
            else:
                inspect(out, idx_true, os.path.dirname(save_model_path), current_epoch, s_batch, att_probs,
                        inv_output_idx, data, args, log)
            n_inspect += 1
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


def inspect(out, idx_true, fig_dir, current_epoch, n, att_probs, inv_output_idx, data, args, log):
    # take only the 1st instance from batch:
    # attention prob distribution
    assert not args.shuffle
    inst_id = data[n][5]
    att = att_probs[0].detach().cpu().numpy()
    log.info("\n{}\nQuery:\n{}".format(inst_id, " ".join(data[n][1])))
    log.info("\nPassage sentence with max. attention:\n{}\n".format(" ".join(data[n][0][np.argmax(att)])))
    plt.plot(att)
    lens = np.array([len(l) for l in data[n][0]])
    plt.plot(lens/np.sum(lens), linestyle='dashed')
    plt.axvline(x=len(data[n][0]), color="red")
    fig_path = "{}/{}_ep{}.png".format(fig_dir, inst_id, current_epoch)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close("all")
    # top k probs and answer ids
    out_probs, out_i = torch.topk(torch.exp(out[0]), 10)
    out_ans = [inv_output_idx[i.item()] for i in out_i]
    log.info("Gold answer: {}".format(inv_output_idx[idx_true[0].item()]))
    log.info("Predicted (k-best):")
    log.info("___________________")
    for a, p in zip(out_ans, list(out_probs.detach().cpu().numpy())):
        log.info("{}\t{}".format(a, p))


def inspect_kv(out, idx_true, fig_dir, current_epoch, n, att_probs, inv_output_idx, data, args, log):
    # take only the 1st instance from batch:
    # attention prob distribution
    assert not args.shuffle
    inst_id = data[n][5]
    att = att_probs[0].detach().cpu().numpy()
    log.info("\n{}\nQuery:\n{}".format(inst_id, " ".join(data[n][1])))
    log.info("\nPassage sentence with max. attention:\n{}\n".format(" ".join(data[n][0][0][np.argmax(att)])))
    plt.plot(att)
    lens = np.array([len(l) for l in data[n][0][0]])
    plt.plot(lens/np.sum(lens), linestyle='dashed')
    plt.axvline(x=len(data[n][0][0]), color="red")
    fig_path = "{}/{}_ep{}.png".format(fig_dir, inst_id, current_epoch)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close("all")
    # top k probs and answer ids
    out_probs, out_i = torch.topk(torch.exp(out[0]), 10)
    out_ans = [inv_output_idx[i.item()] for i in out_i]
    log.info("Gold answer: {}".format(inv_output_idx[idx_true[0].item()]))
    log.info("Predicted (k-best):")
    log.info("___________________")
    for a, p in zip(out_ans, list(out_probs.detach().cpu().numpy())):
        log.info("{}\t{}".format(a, p))


def main():
    arg_parser = argparse.ArgumentParser(description="parser for End-to-End Memory Networks")

    arg_parser.add_argument("--anneal-epoch", type=int, default=25,
                            help="anneal every [anneal-epoch] epoch, default: 25")
    arg_parser.add_argument("--anneal-factor", type=int, default=2,
                            help="factor to anneal by every 'anneal-epoch(s)', default: 2")
    arg_parser.add_argument("--average_embs", type=int, default=1, help="Flag to average context embs instead of summing.")
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
    if args.train == 0 and args.eval == 1:
        logdir = os.path.dirname(args.load_model_path)
        log = get_logger(logdir + "/log_eval")
    else:
        logdir = "{}{}".format(exp_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log = get_logger(logdir + "/log")
    #if args.inspect:
    #    log_inspect = get_logger(logdir + "/inspect")
    #else:
    #    log_inspect = None

    for argk, argv in sorted(vars(args).items()):
        log.info("{}: {}".format(argk, argv))
    log.info("")
    print("Output to {}".format(logdir))
    save_model_path = model_path(logdir, args)

    if args.dataset == "clicr":
        if args.mode == "standard":
            # load data
            data, val_data, test_data, sentence_size, vocab_size, story_size, word_idx, output_size, output_idx = process_data_clicr(args, log=log)
            if args.pretrained_word_embed:
                log.info("Using pretrained word embeddings: {}".format(args.pretrained_word_embed))
            else:
                log.info("Using random initialization.")
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
                train_network(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx,
                              sentence_size, story_size=story_size,
                              vocab_size=vocab_size, output_size=output_size, output_idx=output_idx, save_model_path=save_model_path, args=args, log=log)
            if args.eval == 1:
                if args.train == 1:
                    model = save_model_path
                else:
                    model = args.load_model_path
                eval_network(vocab_size, story_size, sentence_size, model, word_idx, output_size, output_idx, test_batches_id, test_data, log, logdir, args, cuda=args.cuda, test_q_ids=test_q_ids, ignore_missing_preds=args.ignore_missing_preds)
        elif args.mode == "kv":
            # load data
            data, val_data, test_data, k_size, v_size, vocab_size, story_size, word_idx, output_size, output_idx = process_data_clicr_kv(args, log=log)
            if args.pretrained_word_embed:
                log.info("Using pretrained word embeddings: {}".format(args.pretrained_word_embed))
            else:
                log.info("Using random initialization.")
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
                train_network(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx,
                              k_size, story_size=story_size,
                              vocab_size=vocab_size, output_size=output_size, output_idx=output_idx, save_model_path=save_model_path, args=args, log=log)
            if args.eval == 1:
                if args.train == 1:
                    model = save_model_path
                else:
                    model = args.load_model_path
                eval_network(vocab_size, story_size, k_size, model, word_idx, output_size, output_idx, test_batches_id, test_data, log, logdir, args, cuda=args.cuda, test_q_ids=test_q_ids, ignore_missing_preds=args.ignore_missing_preds)



    elif args.dataset == "babi":
        data, test_data, sentence_size, vocab_size, story_size, word_idx = process_data(args)
        # get batch indices
        # TODO: don't leave out instances
        n_train = len(data)
        n_test = len(test_data)
        train_batches_id = list(zip(range(0, n_train - args.batch_size, args.batch_size),
                                    range(args.batch_size, n_train, args.batch_size)))
        test_batches_id = list(zip(range(0, n_test - args.batch_size, args.batch_size),
                                   range(args.batch_size, n_test, args.batch_size)))

        if args.train == 1:
            print("dbg: for babi val=test")
            train_network(train_batches_id, test_batches_id, test_batches_id, data, test_data, test_data, word_idx,
                          sentence_size, story_size=story_size,
                          vocab_size=vocab_size, save_model_path=save_model_path, args=args, log=log)

        if args.eval == 1:
            if args.train == 1:
                model = save_model_path
            else:
                model = args.load_model_path
            eval_network(vocab_size, story_size, sentence_size, model, word_idx, test_batches_id, test_data, log, logdir, args, cuda=args.cuda)
    else:
        raise ValueError


if __name__ == '__main__':
    main()
