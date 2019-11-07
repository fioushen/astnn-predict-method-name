import torch.nn as nn
import torch


class TreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(TreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(torch.zeros(size, self.encode_dim))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, self.th.LongTensor(index),
                                                          self.embedding(self.th.LongTensor(current_node))))

        for c in range(len(children)):
            zeros = self.create_tensor(torch.zeros(size, self.encode_dim))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, self.th.LongTensor(children_index[c]), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = self.th.LongTensor(batch_index)
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(torch.zeros(self.batch_size, self.encode_dim))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class ASTNN(nn.Module):
    def __init__(self, output_dim, num_embeddings, embedding_dim, embeddings, batch_size,
                 hidden_dim=100, encode_dim=128, use_gpu=True):
        super(ASTNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = num_embeddings
        self.encode_dim = encode_dim

        self.encoder = TreeEncoder(self.vocab_size, embedding_dim, self.encode_dim,
                                   self.batch_size, self.gpu, embeddings)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, output_dim)

    def init_hidden(self):
        if self.gpu is True:
            return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda()
        else:
            return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)

    def get_zeros(self, num):
        zeros = torch.zeros(num, self.encode_dim)
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = self.encoder([tree for tree_seq in x for tree in tree_seq], sum(lens))

        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)

        # gru
        gru_out, hidden = self.bigru(encodes, self.init_hidden())

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = torch.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        # linear
        y = self.hidden2label(gru_out)
        return y

