from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from net_util import masked_log_softmax, masked_softmax, masked_sigmoid
from util import get_position_encoding, long_tensor_type, load_emb, float_tensor_type, load_output_emb


class N2N(torch.nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hops, story_size, args, word_idx, output_size, output_idx):
        super(N2N, self).__init__()

        self.embed_size = embed_size
        self.batch_size = batch_size
        self.story_size = story_size
        self.hops = hops
        self.pretrained_word_embed = args.pretrained_word_embed
        self.pretrained_output_layer = args.pretrained_output_layer
        self.freeze_pretrained_word_embed = args.freeze_pretrained_word_embed
        self.word_idx = word_idx
        self.output_idx = output_idx
        self.att_type = args.att_type
        self.args = args

        if self.hops <= 0:
            raise ValueError("Number of hops have to be greater than 0")

        if self.hops > 3:
            raise ValueError("Number of hops should be less than 4")

        # story embedding
        if args.pretrained_word_embed:
            self.A1, dim = load_emb(args.pretrained_word_embed, self.word_idx, freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.A1 = nn.Embedding(vocab_size, embed_size)
            self.A1.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))

        """
        # query embedding
        if args.pretrained_word_embed:
            self.B1, dim = load_emb(args.pretrained_word_embed, self.word_idx,
                                    freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.B1 = nn.Embedding(vocab_size, embed_size)
            self.B1.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        # temporal encoding
        # self.TA = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))
        """
        # for 1 hop:
        # for >1 hop:
        """
        if args.pretrained_word_embed:
            self.A2, dim = load_emb(args.pretrained_word_embed, self.word_idx, freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.A2 = nn.Embedding(vocab_size, embed_size)
            self.A2.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        # self.TA2 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))
        """
        """
        # query embedding
        if args.pretrained_word_embed:
            self.B2, dim = load_emb(args.pretrained_word_embed, self.word_idx,
                                    freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.B2 = nn.Embedding(vocab_size, embed_size)
            self.B2.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        """
        if self.hops >= 2:
            if args.pretrained_word_embed:
                self.A3, dim = load_emb(args.pretrained_word_embed, self.word_idx, freeze=args.freeze_pretrained_word_embed)
                assert dim == self.embed_size
            else:
                self.A3 = nn.Embedding(vocab_size, embed_size)
                self.A3.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))

            # self.TA3 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))

        if self.hops >= 3:
            if args.pretrained_word_embed:
                self.A4, dim = load_emb(args.pretrained_word_embed, self.word_idx, freeze=args.freeze_pretrained_word_embed)
                assert dim == self.embed_size
            else:
                self.A4 = nn.Embedding(vocab_size, embed_size)
                self.A4.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
            # self.TA4 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))


        #self.G =nn.Linear(embed_size, embed_size)

        # final weight matrix
        # self.W = nn.Parameter(torch.randn(embed_size, vocab_size), requires_grad=True)
        #self.nonlin = nn.ReLU()
        #self.lin = nn.Linear(embed_size, embed_size)
        #self.dropout = nn.Dropout(0.5)
        #self.lin_bn = nn.BatchNorm1d(4*embed_size)


        if self.att_type == "cosine":
            self.cos = nn.CosineSimilarity(dim=2)
        elif self.att_type == "bilinear":
            self.bil = nn.Bilinear(embed_size, embed_size, embed_size)

        #self.lin = nn.Linear(embed_size*4, embed_size)
        self.lin_final = nn.Linear(embed_size, output_size)
        if self.pretrained_output_layer:
            self.lin_final.weight, _ = load_output_emb(args.pretrained_output_layer, self.output_idx)
        #self.lin_final = nn.Linear(embed_size, output_size)
        #self.lin_final = nn.Linear(embed_size, vocab_size)
        #self.lin_final.weight = nn.Parameter(self.A1.weight)
        #self.lin_final_bn = nn.BatchNorm1d(output_size)
        #self.lin_final_bn = nn.BatchNorm1d(vocab_size)

    def forward(self, trainS, trainQ, trainVM, trainPM, trainSM, trainQM, inspect):
        """
        :param trainVM: a B*V tensor masking all predictions which are not words/entities in the relevant document
        """
        S = Variable(trainS, requires_grad=False)
        Q = Variable(torch.squeeze(trainQ, 1), requires_grad=False)

        queries_emb = self.A1(Q)
        #queries_emb = self.B1(Q)

        position_encoding = get_position_encoding(queries_emb.size(0), queries_emb.size(1), self.embed_size)
        queries = queries_emb * position_encoding
        # zero out the masked (padded) word embeddings:
        queries = queries * trainQM.unsqueeze(2).expand_as(queries)

        queries_rep = torch.sum(queries, dim=1)
        # w_u = queries_sum
        # for i in range(self.hops):
        #     w_u = self.one_hop(S, w_u, self.A[i], self.A[i + 1], self.TA[i], self.TA[i + 1])
        if self.args.average_embs:
            normalizer = torch.sum(trainQM, dim=1).unsqueeze(1).expand_as(queries_rep)
            normalizer[normalizer==0.] = float("Inf")
            queries_rep = queries_rep / normalizer
        if inspect:
            #w_u, att_probs = self.hop(S, queries_rep, self.A1, self.A2, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
            w_u, att_probs = self.hop(S, queries_rep, self.A1, self.A1, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
        else:
            #w_u = self.hop(S, queries_rep, self.A1, self.A2, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
            w_u = self.hop(S, queries_rep, self.A1, self.A1, trainPM, trainSM, inspect)  # , self.TA, self.TA2)

        if self.hops >= 2:
            if inspect:
                #w_u, att_probs = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
                w_u, att_probs = self.hop(S, w_u, self.A3, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
            else:
                #w_u = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
                w_u = self.hop(S, w_u, self.A3, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)

        if self.hops >= 3:
            if inspect:
                #w_u, att_probs = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
                w_u, att_probs = self.hop(S, w_u, self.A4, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
            else:
                #w_u = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
                w_u = self.hop(S, w_u, self.A4, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)

        # wx = torch.mm(w_u, self.W)
        #wx = self.lin_bn(wx)
        #wx = self.nonlin(wx)

        wx = w_u
        #wx = self.dropout(self.lin(wx))
        # wx = self.lin2(w_u)
        # wx = self.nonlin(wx)
        wx = self.lin_final(wx)
        #wx = self.lin_final_bn(wx)

        # Final layer
        y_pred = wx
        # mask for output answers

        #if trainVM is not None:
        #    y_pred = y_pred * trainVM
        #return y_pred

        y_pred_m = trainVM
        #y_pred_m = None
        out = masked_log_softmax(y_pred, y_pred_m)
        if inspect:
            return out, att_probs
        else:
            return out

    def hop(self, trainS, u_k_1, A_k, C_k, PM, SM, inspect):  # , temp_A_k, temp_C_k):
        mem_emb_A = self.embed_story(trainS, A_k, SM)
        mem_emb_C = self.embed_story(trainS, C_k, SM)

        mem_emb_A_temp = mem_emb_A  # + temp_A_k
        mem_emb_C_temp = mem_emb_C  # + temp_C_k

        #u_k_1 = self.G(u_k_1)
        u_k_1_list = [u_k_1] * self.story_size

        queries_temp = torch.squeeze(torch.stack(u_k_1_list, dim=1), 2)
        #probabs = mem_emb_A_temp * queries_temp
        # zero out the masked (padded) sentence embeddings:
        #probabs = probabs * PM.unsqueeze(2).expand_as(probabs)
        probabs = self.cos(mem_emb_A_temp, queries_temp)
        #probabs = masked_softmax(torch.squeeze(torch.sum(probabs, dim=2)), PM)
        probabs = masked_softmax(probabs, PM)
        mem_emb_C_temp = mem_emb_C_temp.permute(0, 2, 1)
        probabs_temp = probabs.unsqueeze(1).expand_as(mem_emb_C_temp)

        pre_w = torch.mul(mem_emb_C_temp, probabs_temp)

        o = torch.sum(pre_w, dim=2)

        #u_k = torch.squeeze(o) #+ torch.squeeze(u_k_1)

        if inspect:
            return torch.cat((o, u_k_1, o+u_k_1, o*u_k_1), dim=1), probabs
        else:
            return torch.cat((o, u_k_1, o+u_k_1, o*u_k_1), dim=1)

    def embed_story(self, story_batch, embedding_layer, sent_mask, positional=True):
        story_embedding_list = []
        if positional:
            position_encoding = get_position_encoding(story_batch.size()[1], story_batch.size()[2], self.embed_size)
        else:
            position_encoding = None

        for story in story_batch.split(1):
            story_variable = Variable(torch.squeeze(story, 0).data.type(long_tensor_type))
            story_embedding = embedding_layer(story_variable)
            if position_encoding is not None:
                story_embedding = story_embedding * position_encoding
            story_embedding_list.append(story_embedding)

        batch_story_embedding_temp = torch.stack(story_embedding_list)
        # zero out the masked (padded) word embeddings in the passage:
        batch_story_embedding_temp = batch_story_embedding_temp * sent_mask.unsqueeze(3).expand_as(batch_story_embedding_temp)
        batch_story_embedding = torch.sum(batch_story_embedding_temp, dim=2)
        if self.args.average_embs:
            normalizer = torch.sum(sent_mask, dim=2).unsqueeze(2).expand_as(batch_story_embedding)
            normalizer[normalizer==0.] = float("Inf")
            batch_story_embedding = batch_story_embedding / normalizer

        return torch.squeeze(batch_story_embedding, dim=2)


class KVN2N(N2N):
    def forward(self, trainK, trainV, trainQ, trainVM, trainPM, trainKM, trainQM, inspect, positional=True, multi_att_supervision=False):
        """
        :param trainVM: a B*V tensor masking all predictions which are not words/entities in the relevant document
        """
        K = Variable(trainK, requires_grad=False)
        V = Variable(trainV, requires_grad=False)
        Q = Variable(torch.squeeze(trainQ, 1), requires_grad=False)

        queries = self.A1(Q)
        #queries_emb = self.B1(Q)

        if positional:
            position_encoding = get_position_encoding(queries.size(0), queries.size(1), self.embed_size)
            queries = queries * position_encoding
        # zero out the masked (padded) word embeddings:
        queries = queries * trainQM.unsqueeze(2).expand_as(queries)

        queries_rep = torch.sum(queries, dim=1)
        # w_u = queries_sum
        # for i in range(self.hops):
        #     w_u = self.one_hop(S, w_u, self.A[i], self.A[i + 1], self.TA[i], self.TA[i + 1])
        if self.args.average_embs:
            normalizer = torch.sum(trainQM, dim=1).unsqueeze(1).expand_as(queries_rep)
            normalizer[normalizer==0.] = float("Inf")
            queries_rep = queries_rep / normalizer

        if inspect:
            #w_u, att_probs = self.hop(S, queries_rep, self.A1, self.A2, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
            w_u, att_probs = self.hop(K, V, queries_rep, self.A1, self.A1, trainPM, trainKM, inspect, positional=positional, multi_att_supervision=multi_att_supervision)  # , self.TA, self.TA2)
        else:
            #w_u = self.hop(S, queries_rep, self.A1, self.A2, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
            w_u = self.hop(K, V, queries_rep, self.A1, self.A1, trainPM, trainKM, inspect, positional=positional)  # , self.TA, self.TA2)

        if self.hops >= 2:
            if inspect:
                #w_u, att_probs = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
                w_u, att_probs = self.hop(K, V, w_u, self.A3, self.A3, trainPM, trainKM, inspect, positional=positional, multi_att_supervision=multi_att_supervision)  # , self.TA, self.TA3)
            else:
                #w_u = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
                w_u = self.hop(K, V, w_u, self.A3, self.A3, trainPM, trainKM, inspect, positional=positional)  # , self.TA, self.TA3)

        if self.hops >= 3:
            if inspect:
                #w_u, att_probs = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
                w_u, att_probs = self.hop(K, V, w_u, self.A4, self.A4, trainPM, trainKM, inspect, positional=positional, multi_att_supervision=multi_att_supervision)  # , self.TA, self.TA4)
            else:
                #w_u = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
                w_u = self.hop(K, V, w_u, self.A4, self.A4, trainPM, trainKM, inspect, positional=positional)  # , self.TA, self.TA4)

        # wx = torch.mm(w_u, self.W)


        #wx = self.lin_bn(wx)
        #wx = self.nonlin(wx)

        wx = w_u
        #wx = self.dropout(self.lin(wx))
        #wx = self.lin_bn(self.lin(w_u))
        #wx = self.nonlin(wx)
        #wx = self.dropout(wx)
        wx = self.lin_final(wx)
        #wx = self.lin_final_bn(wx)

        # Final layer
        y_pred = wx
        # mask for output answers

        #if trainVM is not None:
        #    y_pred = y_pred * trainVM
        #return y_pred

        y_pred_m = trainVM
        #y_pred_m = None
        out = masked_log_softmax(y_pred, y_pred_m)
        if inspect:
            return out, att_probs
        else:
            return out

    def hop(self, trainK, trainV, u_k_1, A_k, C_k, PM, KM, inspect, positional=True, multi_att_supervision=True):  # , temp_A_k, temp_C_k):
        mem_emb_A = self.embed_story(trainK, A_k, KM, positional=positional)  # B*S*d
        mem_emb_C = self.embed_values(trainV, C_k)  # B*S*d

        mem_emb_A_temp = mem_emb_A  # + temp_A_k
        mem_emb_C_temp = mem_emb_C  # + temp_C_k

        #u_k_1 = self.G(u_k_1)
        u_k_1_list = [u_k_1] * self.story_size

        queries_temp = torch.squeeze(torch.stack(u_k_1_list, dim=1), 2)
        #probabs = mem_emb_A_temp * queries_temp
        # zero out the masked (padded) sentence embeddings:
        #probabs = probabs * PM.unsqueeze(2).expand_as(probabs)
        #probabs = self.cos(mem_emb_A_temp, queries_temp)  # B*S
        if self.att_type == "cosine":
            probabs = self.cos(mem_emb_A_temp, queries_temp)  # B*S
        elif self.att_type =="bilinear":
            probabs = self.bil(mem_emb_A_temp, queries_temp)  # B*S*d
            probabs = torch.sum(probabs, dim=2) # B*S
        if multi_att_supervision:
            probabs = masked_sigmoid(probabs, PM)
        else:
            probabs_log = masked_log_softmax(probabs, PM)  # B*S
            probabs = torch.exp(probabs_log)
        mem_emb_C_temp = mem_emb_C_temp.permute(0, 2, 1)   # B*d*S
        probabs_temp = probabs.unsqueeze(1).expand_as(mem_emb_C_temp)

        pre_w = torch.mul(mem_emb_C_temp, probabs_temp)
        o = torch.sum(pre_w, dim=2)

        #u_k = torch.squeeze(o) #+ torch.squeeze(u_k_1)

        #hop_o = torch.cat((o, u_k_1, o + u_k_1, o * u_k_1), dim=1)  # B*4d
        hop_o = o + u_k_1  # B*d
        if inspect:
            return hop_o, probabs if multi_att_supervision else probabs_log

        else:
            return hop_o


    def embed_values(self, val_batch, embedding_layer):
        vals_variable = Variable(val_batch.data.type(long_tensor_type))

        return embedding_layer(vals_variable)

class KVAtt(torch.nn.Module):
    """
    A key-value attention ~ max. embedding similarity between q and p
    """
    def __init__(self, batch_size, embed_size, vocab_size, story_size, args, word_idx, output_size):
        super(KVAtt, self).__init__()

        self.embed_size = embed_size
        self.batch_size = batch_size
        self.story_size = story_size
        self.pretrained_word_embed = args.pretrained_word_embed
        self.freeze_pretrained_word_embed = args.freeze_pretrained_word_embed
        self.word_idx = word_idx
        self.args = args
        self.output_size = output_size

        # story embedding
        if args.pretrained_word_embed:
            self.A1, dim = load_emb(args.pretrained_word_embed, self.word_idx,
                                    freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.A1 = nn.Embedding(vocab_size, embed_size)
            self.A1.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))

        self.cos = nn.CosineSimilarity(dim=2)


    def forward(self, trainK, trainV, trainQ, trainVM, trainPM, trainKM, trainQM, inspect, positional=True, attention_sum=False):
        """
        :param trainVM: a B*V tensor masking all predictions which are not words/entities in the relevant document
        """
        K = Variable(trainK, requires_grad=False)
        Q = Variable(torch.squeeze(trainQ, 1), requires_grad=False)

        queries = self.A1(Q)

        if positional:
            position_encoding = get_position_encoding(queries.size(0), queries.size(1), self.embed_size)
            queries = queries * position_encoding
        # zero out the masked (padded) word embeddings:
        queries = queries * trainQM.unsqueeze(2).expand_as(queries)

        queries_rep = torch.sum(queries, dim=1)
        if self.args.average_embs:
            normalizer = torch.sum(trainQM, dim=1).unsqueeze(1).expand_as(queries_rep)
            normalizer[normalizer==0.] = float("Inf")
            queries_rep = queries_rep / normalizer

        att_scores = self.attention(K, queries_rep, self.A1, trainKM, positional=positional)  # , self.TA, self.TA2)
        # probs over keys
        att_probs = masked_log_softmax(att_scores, trainPM)
        if attention_sum:
            probs_out, val_idx = self.max_of_attention_sum(trainV, att_probs)
        else:
            probs_out, idx_out = torch.max(att_probs, 1)
            # get ids for values
            val_idx = trainV[range(self.batch_size), idx_out]
        # initialize y to very small number (log space)
        y = Variable(torch.full((self.batch_size, self.output_size), -100.), requires_grad=False).type(float_tensor_type)
        y[range(self.batch_size), val_idx] = probs_out

        return y, val_idx, att_probs

    def max_of_attention_sum(self, trainV, att_probs):
        probs_out = []
        idx_out = []
        i_len, j_len = trainV.shape
        for i in range(i_len):
            d = defaultdict(float)
            for j in range(j_len):
                ent_idx = trainV[i,j]
                if ent_idx == 0:
                    continue
                d[ent_idx] += torch.exp(att_probs[i,j])
            max_ent, max_prob = sorted(d.items(), key=lambda x:x[1], reverse=True)[0]
            probs_out.append(torch.log(max_prob))
            idx_out.append(max_ent)

        return float_tensor_type(probs_out), long_tensor_type(idx_out)


    def attention(self, trainK, u_k_1, A_k, KM, positional=True):  # , temp_A_k, temp_C_k):
        mem_emb_A = self.embed_story(trainK, A_k, KM, positional=positional)  # B*S*d
        mem_emb_A_temp = mem_emb_A  # + temp_A_k
        u_k_1_list = [u_k_1] * self.story_size
        queries_temp = torch.squeeze(torch.stack(u_k_1_list, dim=1), 2)
        att_scores = self.cos(mem_emb_A_temp, queries_temp)  # B*S

        return att_scores

    def embed_story(self, story_batch, embedding_layer, sent_mask, positional=True):
        story_embedding_list = []
        if positional:
            position_encoding = get_position_encoding(story_batch.size()[1], story_batch.size()[2], self.embed_size)
        else:
            position_encoding = None

        for story in story_batch.split(1):
            story_variable = Variable(torch.squeeze(story, 0).data.type(long_tensor_type))
            story_embedding = embedding_layer(story_variable)
            if position_encoding is not None:
                story_embedding = story_embedding * position_encoding
            story_embedding_list.append(story_embedding)

        batch_story_embedding_temp = torch.stack(story_embedding_list)
        # zero out the masked (padded) word embeddings in the passage:
        batch_story_embedding_temp = batch_story_embedding_temp * sent_mask.unsqueeze(3).expand_as(batch_story_embedding_temp)
        batch_story_embedding = torch.sum(batch_story_embedding_temp, dim=2)
        if self.args.average_embs:
            normalizer = torch.sum(sent_mask, dim=2).unsqueeze(2).expand_as(batch_story_embedding)
            normalizer[normalizer==0.] = float("Inf")
            batch_story_embedding = batch_story_embedding / normalizer

        return torch.squeeze(batch_story_embedding, dim=2)

