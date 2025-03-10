import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.linear_q = nn.Linear(embed_size, embed_size, bias=False)
        self.linear_k = nn.Linear(embed_size, embed_size, bias=False)
        self.linear_v = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings, edge_index):
        device = embeddings.device
        edge_index = edge_index.to(device)

        queries = self.linear_q(embeddings).view(-1, self.num_heads, self.head_dim)
        keys = self.linear_k(embeddings).view(-1, self.num_heads, self.head_dim)
        values = self.linear_v(embeddings).view(-1, self.num_heads, self.head_dim)

        edge_q = queries[edge_index[0]]
        edge_k = keys[edge_index[1]]
        edge_v = values[edge_index[1]]

        attention_scores = (edge_q * edge_k).sum(dim=-1) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=0)

        weighted_values = edge_v * attention_weights.unsqueeze(-1) 
        out = torch.zeros_like(values, device=device)
        out.index_add_(0, edge_index[0], weighted_values)

        out = out.view(-1, self.num_heads * self.head_dim)
        return self.fc_out(out)

class AttentionLayer(nn.Module):
    def __init__(self, embed_size):
        super(AttentionLayer, self).__init__()
        self.linear_q = nn.Linear(embed_size, embed_size, bias=False)
        self.linear_k = nn.Linear(embed_size, embed_size, bias=False)
        self.linear_v = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, embeddings, edge_index):
        device = embeddings.device
        edge_index = edge_index.to(device)

        queries = self.linear_q(embeddings)
        keys = self.linear_k(embeddings)
        values = self.linear_v(embeddings)

        edge_queries = queries[edge_index[0]]
        edge_keys = keys[edge_index[1]]
        edge_values = values[edge_index[1]]

        attention_scores = (edge_queries * edge_keys).sum(dim=-1) / (embeddings.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=0)

        weighted_values = edge_values * attention_weights.unsqueeze(-1)

        out_embeddings = torch.zeros_like(embeddings, device=device) 
        out_embeddings.index_add_(0, edge_index[0], weighted_values)

        return out_embeddings

class GraphNCF_A(nn.Module):
    def __init__(self,
                 norm_laplacian: sp.coo_matrix,
                 eye_matrix: sp.coo_matrix,
                 num_user: int,
                 num_item: int,
                 embed_size: int,
                 node_dropout_ratio: float,
                 layer_size: int,
                 device,
                 num_heads=4,
                 mess_dropout=[0.1, 0.1, 0.1]):
        super(GraphNCF_A, self).__init__()
        self.device = device
        self.num_users = num_user
        self.num_items = num_item
        self.embed_size = embed_size
        self.node_dropout_ratio = node_dropout_ratio
        self.mess_dropout = mess_dropout
        self.layer_size = layer_size

        self.embedding_user = nn.Embedding(self.num_users, self.embed_size)
        self.embedding_item = nn.Embedding(self.num_items, self.embed_size)

        self.attention_layer = MultiHeadAttentionLayer(self.embed_size, num_heads=num_heads)

        layer_list = []
        for _ in range(self.layer_size):
            layer_list.append(nn.Linear(self.embed_size, self.embed_size))
        self.layer1 = nn.Sequential(*layer_list)
        self.layer2 = nn.Sequential(*layer_list)

        self._init_weight()

        self.norm_laplacian = self._convert_mat2tensor(norm_laplacian).to(self.device)
        self.eye_matrix = self._convert_mat2tensor(eye_matrix).to(self.device)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        for layer in self.layer1:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.01)

        for layer in self.layer2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.01)

    def _node_dropout(self, mat):
        node_mask = nn.Dropout(self.node_dropout_ratio)(
            torch.ones(mat._nnz(), device=self.device)
        ).bool()
        indices = mat._indices()
        values = mat._values()
        indices = indices[:, node_mask]
        values = values[node_mask]
        out = torch.sparse_coo_tensor(indices, values, mat.shape, device=self.device)
        return out

    def _convert_mat2tensor(self, mat):
        indices = torch.LongTensor([mat.row, mat.col]).to(self.device)
        values = torch.tensor(mat.data, dtype=torch.float).to(self.device)
        return torch.sparse_coo_tensor(indices, values, mat.shape, device=self.device)

    def forward(self, users, pos_items, neg_items, edge_index=None, edge_weight=None, use_dropout=True):
        if use_dropout:
            norm_laplacian = self._node_dropout(self.norm_laplacian)
        else:
            norm_laplacian = self.norm_laplacian

        if edge_index is not None and edge_weight is not None:
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)

            edge_tensor = torch.sparse_coo_tensor(edge_index, edge_weight, norm_laplacian.shape, device=self.device)
            norm_laplacian = edge_tensor + self.eye_matrix
        else:
            norm_laplacian += self.eye_matrix

        prev_embedding = torch.cat((self.embedding_user.weight, self.embedding_item.weight), dim=0)
        all_embedding = [prev_embedding]

        for index, (l1, l2) in enumerate(zip(self.layer1, self.layer2)):
            first_term = torch.sparse.mm(norm_laplacian, prev_embedding)
            first_term = self.attention_layer(first_term, edge_index)
            first_term = torch.matmul(first_term, l1.weight) + l1.bias

            second_term = prev_embedding * prev_embedding
            second_term = torch.sparse.mm(norm_laplacian, second_term)
            second_term = self.attention_layer(second_term, edge_index)
            second_term = torch.matmul(second_term, l2.weight) + l2.bias

            prev_embedding = nn.LeakyReLU(negative_slope=0.2)(first_term + second_term)
            prev_embedding = nn.Dropout(self.mess_dropout[index])(prev_embedding)
            prev_embedding = F.normalize(prev_embedding, p=2, dim=1)

            all_embedding.append(prev_embedding)

        all_embedding = torch.cat(all_embedding, dim=1)
        self.user_embeddings = all_embedding[:self.num_users, :]
        self.item_embeddings = all_embedding[self.num_users:, :]

        users_embed = self.user_embeddings[users, :]
        pos_item_embeddings = self.item_embeddings[pos_items, :]
        neg_item_embeddings = self.item_embeddings[neg_items, :]

        return users_embed, pos_item_embeddings, neg_item_embeddings