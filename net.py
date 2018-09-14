import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from net_util import masked_log_softmax, masked_softmax, masked_softmin
from util import get_position_encoding, long_tensor_type, load_emb


class N2N(torch.nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hops, story_size, args, word_idx):
        super(N2N, self).__init__()

        self.embed_size = embed_size
        self.batch_size = batch_size
        self.story_size = story_size
        self.hops = hops
        self.pretrained_word_embed = args.pretrained_word_embed
        self.freeze_pretrained_word_embed = args.freeze_pretrained_word_embed
        self.word_idx = word_idx

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
        if args.pretrained_word_embed:
            self.A2, dim = load_emb(args.pretrained_word_embed, self.word_idx, freeze=args.freeze_pretrained_word_embed)
            assert dim == self.embed_size
        else:
            self.A2 = nn.Embedding(vocab_size, embed_size)
            self.A2.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        # self.TA2 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))
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

        # final weight matrix
        # self.W = nn.Parameter(torch.randn(embed_size, vocab_size), requires_grad=True)
        #self.nonlin = nn.Tanh()
        #self.lin = nn.Linear(embed_size, embed_size)
        #self.lin_bn = nn.BatchNorm1d(embed_size)
        self.lin_final = nn.Linear(embed_size, vocab_size)
        self.lin_final_bn = nn.BatchNorm1d(vocab_size)

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

        queries_sum = torch.sum(queries, dim=1)
        # w_u = queries_sum
        # for i in range(self.hops):
        #     w_u = self.one_hop(S, w_u, self.A[i], self.A[i + 1], self.TA[i], self.TA[i + 1])

        if inspect:
            w_u, att_probs = self.hop(S, queries_sum, self.A1, self.A2, trainPM, trainSM, inspect)  # , self.TA, self.TA2)
        else:
            w_u = self.hop(S, queries_sum, self.A1, self.A2, trainPM, trainSM,
                                      inspect)  # , self.TA, self.TA2)

        if self.hops >= 2:
            if inspect:
                w_u, att_probs = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)
            else:
                w_u = self.hop(S, w_u, self.A2, self.A3, trainPM, trainSM, inspect)  # , self.TA, self.TA3)

        if self.hops >= 3:
            if inspect:
                w_u, att_probs = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)
            else:
                w_u = self.hop(S, w_u, self.A3, self.A4, trainPM, trainSM, inspect)  # , self.TA, self.TA4)

        # wx = torch.mm(w_u, self.W)

        #wx = self.lin(w_u)
        #wx = self.lin_bn(wx)
        #wx = self.nonlin(wx)

        wx = w_u
        # wx = self.lin2(w_u)
        # wx = self.nonlin(wx)
        wx = self.lin_final(wx)
        wx = self.lin_final_bn(wx)

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

        u_k_1_list = [u_k_1] * self.story_size

        queries_temp = torch.squeeze(torch.stack(u_k_1_list, dim=1), 2)
        probabs = mem_emb_A_temp * queries_temp
        # zero out the masked (padded) sentence embeddings:
        probabs = probabs * PM.unsqueeze(2).expand_as(probabs)
        #probabs = F.softmax(torch.squeeze(torch.sum(probabs, dim=2)), dim=1)
        probabs = masked_softmax(torch.squeeze(torch.sum(probabs, dim=2)), PM)
        mem_emb_C_temp = mem_emb_C_temp.permute(0, 2, 1)
        probabs_temp = probabs.unsqueeze(1).expand_as(mem_emb_C_temp)

        pre_w = torch.mul(mem_emb_C_temp, probabs_temp)

        o = torch.sum(pre_w, dim=2)

        u_k = torch.squeeze(o) + torch.squeeze(u_k_1)

        if inspect:
            return u_k, probabs
        else:
            return u_k

    def embed_story(self, story_batch, embedding_layer, sent_mask):
        story_embedding_list = []
        position_encoding = get_position_encoding(story_batch.size()[1], story_batch.size()[2], self.embed_size)

        for story in story_batch.split(1):
            story_variable = Variable(torch.squeeze(story, 0).data.type(long_tensor_type))
            story_embedding = embedding_layer(story_variable)
            story_embedding = story_embedding * position_encoding
            story_embedding_list.append(story_embedding)

        batch_story_embedding_temp = torch.stack(story_embedding_list)
        # zero out the masked (padded) word embeddings in the passage:
        batch_story_embedding_temp * sent_mask.unsqueeze(3).expand_as(batch_story_embedding_temp)
        batch_story_embedding = torch.sum(batch_story_embedding_temp, dim=2)

        return torch.squeeze(batch_story_embedding, dim=2)
