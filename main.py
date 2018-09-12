import argparse
from datetime import datetime
import os

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from logger import get_logger
from net import N2N
from util import long_tensor_type, vectorize_data_clicr, vectorized_batches, vectorize_data, evaluate_clicr, save_json, \
    load_clicr, get_q_ids_clicr
from util import process_data, process_data_clicr, get_batch_from_batch_list


def train_network(train_batches_id, val_batches_id, test_batches_id, data, val_data, test_data, word_idx, sentence_size,
                  vocab_size, story_size, save_model_path, args, log):
    net = N2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args, word_idx=word_idx)
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
        vectorizer = vectorize_data_clicr
    elif args.dataset == "babi":
        vectorizer = vectorize_data
    else:
        raise NotImplementedError

    running_loss = 0.0
    best_val_acc_yet = 0.0
    for current_epoch in range(args.epochs):
        train_batch_gen = vectorized_batches(train_batches_id, data, word_idx, sentence_size, story_size, vectorizer, shuffle=args.shuffle)
        current_len = 0
        current_correct = 0
        if current_epoch == 9:
            print()
        for batch, n in zip(train_batch_gen, train_batches_id):
            idx_out, idx_true, out = epoch(batch, net)
            loss = criterion(out, idx_true)
            loss.backward()

            clip_grad_norm(net.parameters(), 40)
            running_loss += loss
            current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
            optimizer.step()
            optimizer.zero_grad()
            # print("Batch {}/{}.".format(n, len(train_batches_id)))

        if current_epoch % args.log_epochs == 0:
            accuracy = 100 * (current_correct / current_len)
            val_acc = calculate_loss_and_accuracy(net, val_batches_id, val_data, word_idx, sentence_size, story_size,
                                                  vectorizer)
            log.info("Epochs: {}, Train Accuracy: {}, Loss: {}, Val_Acc:{}".format(current_epoch, accuracy,
                                                                                running_loss.item(),
                                                                                val_acc))
            if best_val_acc_yet <= val_acc and args.save_model:
                torch.save(net.state_dict(), save_model_path)
                best_val_acc_yet = val_acc

        if current_epoch % args.anneal_epoch == 0 and current_epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / args.anneal_factor
        running_loss = 0.0


def epoch(batch, net):
    story_batch = batch[0]
    query_batch = batch[1]
    answer_batch = batch[2]
    vocabmask_batch = batch[3]

    A = Variable(torch.stack(answer_batch, dim=0), requires_grad=False).type(long_tensor_type)
    _, idx_true = torch.max(A, 1)
    idx_true = torch.squeeze(idx_true)

    S = torch.stack(story_batch, dim=0)
    Q = torch.stack(query_batch, dim=0)
    if vocabmask_batch is not None:
        VM = torch.stack(vocabmask_batch, dim=0)
    else:
        VM = None
    out = net(S, Q, VM)

    _, idx_out = torch.max(out, 1)
    return idx_out, idx_true, out


def update_counts(current_correct, current_len, idx_out, idx_true):
    batch_len, correct = count_predictions(idx_true, idx_out)
    current_len += batch_len
    current_correct += correct
    return current_correct, current_len


def count_predictions(labels, predicted):
    batch_len = len(labels)
    correct = float((predicted == labels).sum())
    return batch_len, correct


def calculate_loss_and_accuracy(net, batches_id, data, word_idx, sentence_size, story_size, vectorizer):
    batch_gen = vectorized_batches(batches_id, data, word_idx, sentence_size, story_size, vectorizer)
    current_len = 0
    current_correct = 0
    for batch in batch_gen:
        idx_out, idx_true, out = epoch(batch, net)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    return 100 * (current_correct / current_len)


def eval_network(vocab_size, story_size, sentence_size, model, word_idx, test_batches_id, test, log, logdir, args, cuda=0., test_q_ids=None):
    log.info("Evaluating")
    net = N2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size, args=args, word_idx=word_idx)
    net.load_state_dict(torch.load(model))
    if torch.cuda.is_available() and cuda == 1:
        net = net.cuda()
    if args.dataset == "clicr":
        vectorizer = vectorize_data_clicr
    elif args.dataset == "babi":
        vectorizer = vectorize_data
    else:
        raise NotImplementedError
    test_batch_gen = vectorized_batches(test_batches_id, test, word_idx, sentence_size, story_size, vectorizer)
    current_len = 0
    current_correct = 0
    preds = {} if args.dataset == "clicr" else None
    inv_word_idx = {v: k for k, v in word_idx.items()}

    for batch, (s_batch, _) in zip(test_batch_gen, test_batches_id):
        idx_out, idx_true, out = epoch(batch, net)
        if preds is not None:
            for c, i in enumerate(idx_out):
                # {query_id: answer}
                preds[test[s_batch+c][5]] = inv_word_idx[i.item()]

        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)

    # produce dummy predictions for query ids that were not classified by the model
    missing = test_q_ids - preds.keys()
    log.info("{} predictions missing out of {}.".format(len(missing), len(test_q_ids)))
    for q_id in missing:
        preds[q_id] = ""

    if preds is not None and args.dataset == "clicr":
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
    arg_parser.add_argument("--joint-training", type=int, default=0, help="joint training flag, default: 0")
    arg_parser.add_argument("--load-model-path", type=str)
    arg_parser.add_argument("--log-epochs", type=int, default=4,
                            help="Number of epochs after which to log progress, default: 4")
    arg_parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default: 0.01")
    arg_parser.add_argument("--max-n-load", type=int, help="maximum number of clicr documents to use, for debugging")
    arg_parser.add_argument("--memory-size", type=int, default=50, help="upper limit on memory size, default: 50")
    arg_parser.add_argument("--pretrained-word-embed", type=str,
                            help="path to the txt file with word embeddings")  # "/nas/corpora/accumulate/clicr/embeddings/4bfb98c2-688e-11e7-aa74-901b0e5592c8/embeddings"
    arg_parser.add_argument("--save-model", action="store_true")
    arg_parser.add_argument("--shuffle", action="store_true")
    arg_parser.add_argument("--task-number", type=int, default=1, help="Babi task to process, default: 1")
    arg_parser.add_argument("--train", type=int, default=1)

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
    for argk, argv in sorted(vars(args).items()):
        log.info("{}: {}".format(argk, argv))
    log.info("")
    print("Output to {}".format(logdir))
    save_model_path = model_path(logdir, args)

    if args.dataset == "clicr":
        # load data
        data, val_data, test_data, sentence_size, vocab_size, story_size, word_idx = process_data_clicr(args, log=log)
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
                          vocab_size=vocab_size, save_model_path=save_model_path, args=args, log=log)
        if args.eval == 1:
            if args.train == 1:
                model = save_model_path
            else:
                model = args.load_model_path
            eval_network(vocab_size, story_size, sentence_size, model, word_idx, test_batches_id, test_data, log, logdir, args, cuda=args.cuda, test_q_ids=test_q_ids)
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
