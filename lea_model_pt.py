import numpy as np
import torch
from torch import nn
from torch.nn import Module, ParameterDict, Parameter
from LEA_Setting import Settings

args = Settings()

def _convert_sp_mat_to_sp_tensor(arr):
    return torch.sparse_coo_tensor(indices=torch.tensor([arr.nonzero()[0], arr.nonzero()[1]]),
                                   values=torch.tensor(arr.data),
                                   size=torch.Size(arr.shape),
                                   dtype=torch.float32).to(args.device)


class LEA_GCN(Module):
    def __init__(self, n_items_A, n_items_B, n_users, graph_matrix):
        super().__init__()
        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.n_users = n_users
        self.graph_matrix = graph_matrix

        self.embedding_size = args.embedding_size
        self.n_fold = args.n_fold
        self.alpha = args.alpha
        self.layer_size = args.layer_size
        self.beta = args.beta
        self.regular_rate_att = args.regular_rate_att
        self.num_heads = args.num_heads
        self.n_layers = args.num_layers
        self.dim_coefficient = args.dim_coefficient
        self.batch_size = args.batch_size
        self.dropout_rate = args.dropout_rate
        self.keep_prob = args.keep_prob
        self.weight_size = eval(self.layer_size)

        self.all_weights = ParameterDict({
            'user_embedding': Parameter(torch.zeros([self.n_users, self.embedding_size])),
            'item_embedding_A': Parameter(torch.zeros([self.n_items_A, self.embedding_size])),
            'item_embedding_B': Parameter(torch.zeros([self.n_items_B, self.embedding_size])),
            'pos_embedding_A': Parameter(torch.zeros([self.n_items_A, self.embedding_size])),
            'pos_embedding_B': Parameter(torch.zeros([self.n_items_B, self.embedding_size])),
            # parameters for domain A
            'W_att_A': Parameter(torch.zeros([self.embedding_size, self.weight_size[0]], dtype=torch.float32)),
            'b_att_A': Parameter(torch.zeros([1, self.weight_size[0]], dtype=torch.float32)),
            'h_att_A': Parameter(torch.ones([self.weight_size[0], 1], dtype=torch.float32)),
            # parameters for domain B
            'W_att_B': Parameter(torch.zeros([self.embedding_size, self.weight_size[0]], dtype=torch.float32)),
            'b_att_B': Parameter(torch.zeros([1, self.weight_size[0]], dtype=torch.float32)),
            'h_att_B': Parameter(torch.ones([self.weight_size[0], 1], dtype=torch.float32)),
        })
        self._init_weights()

        self.fc1 = nn.Linear(16, self.embedding_size * self.dim_coefficient)
        self.fc2 = nn.Linear(32, self.batch_size // self.dim_coefficient)
        self.fc3 = nn.Linear(64, self.embedding_size * self.dim_coefficient // self.num_heads)
        self.fc4 = nn.Linear(32, self.embedding_size)

        self.fc_a = nn.Linear(80, n_items_A)
        self.fc_b = nn.Linear(80, n_items_B)

    
    def _init_weights(self):
        for k, v in self.all_weights.items():
            if not k.startswith('h_att'):
                torch.nn.init.xavier_uniform_(v)

    def forward(self, inputs):
        uid, seq_A, seq_B, self.len_A, self.len_B, pos_A, pos_B = inputs
        i_embeddings_A, u_embeddings, i_embeddings_B = self.graph_encoder(self.n_items_A, self.n_users, self.n_items_B)

        seq_emb_A_output, seq_emb_B_output ,seq_emb_AB_output = self.seq_encoder(uid, seq_A, seq_B, pos_A, pos_B, i_embeddings_A, u_embeddings, i_embeddings_B)
        pred_A = self.prediction_A( seq_emb_A_output ,seq_emb_AB_output)
        pred_B = self.prediction_B( seq_emb_B_output ,seq_emb_AB_output)
        return pred_A, pred_B
    
    def prediction_A(self,  seq_emb_A_output ,seq_emb_AB_output):
        concat_output = torch.concat([seq_emb_AB_output, seq_emb_A_output], axis=-1)
        concat_output = torch.dropout(concat_output, self.keep_prob, self.training)
        pred_A = self.fc_a(concat_output)
        return pred_A
    
    def prediction_B(self,  seq_emb_B_output ,seq_emb_AB_output):
        concat_output = torch.concat([seq_emb_AB_output, seq_emb_B_output], axis=-1)
        concat_output = torch.dropout(concat_output, self.keep_prob, self.training)
        pred_B = self.fc_b(concat_output)
        return pred_B
    
    def graph_encoder(self, n_items_A, n_users, n_items_B):
        graph_info = self.unzip_laplace(self.graph_matrix)

        ego_embeddings = torch.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                    self.all_weights['item_embedding_B']], axis=0)  # 基本结点表示
        
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(args.n_fold):
                temp_embed.append(torch.sparse.mm(graph_info[f], ego_embeddings))  # 把结点表示加上额外的大阵信息

            # sum messages of neighbors.
            side_embeddings = torch.concat(temp_embed, 0)

            all_embeddings += [side_embeddings]

        all_embeddings = torch.stack(all_embeddings, 1)  # layer-wise aggregation
        all_embeddings = torch.mean(all_embeddings, axis=1, keepdims=False)  # sum the layer aggregation and the normalizer
        g_embeddings_A, u_g_embeddings, g_embeddings_B = torch.split(all_embeddings, [n_items_A, n_users, n_items_B], 0)

        return g_embeddings_A, u_g_embeddings, g_embeddings_B

    def unzip_laplace(self, X):
        unzip_info = []
        fold_len = (X.shape[0]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            unzip_info.append(_convert_sp_mat_to_sp_tensor(X[start:end]))
        return unzip_info

    def seq_encoder(self, uid, seq_A, seq_B, pos_A, pos_B, i_embeddings_A, u_embeddings, i_embeddings_B,
                    ):
        self.user_embed = nn.functional.embedding(uid, u_embeddings)
        
        ##### domain A:
        item_embed_A = nn.functional.embedding(seq_A, i_embeddings_A)
        pos_embed_A = nn.functional.embedding(pos_A, self.all_weights['pos_embedding_A'])
        item_pos_A = item_embed_A + self.alpha * pos_embed_A


        # 2 using EA_channel_1 for collaborative filtering signals
        ext_embed_A1 = self.ext_attention_encoder_1(item_embed_A, is_A=True)

        # 3 using EA_channel_2 for positional encoding and sequential pattern
        ext_embed_A2 = self.ext_attention_encoder_2(item_pos_A, dim_coefficient=self.dim_coefficient)

        self.seq_embed_A = ext_embed_A1 + ext_embed_A2

        seq_emb_A_output = torch.concat([self.seq_embed_A, self.user_embed], axis=1)
        seq_emb_A_output = torch.dropout(seq_emb_A_output, self.dropout_rate, self.training)
        ##### domain B
        item_embed_B = nn.functional.embedding(seq_B, i_embeddings_B)
        pos_embed_B = nn.functional.embedding(pos_B, self.all_weights['pos_embedding_B'])
        item_pos_B = item_embed_B + self.alpha * pos_embed_B
        # 1 simple max_pooling
        # seq_embed_B_state = tf.reduce_max((seq_emb_B_output), 1)

        # 2 using EA_channel_1 for collaborative filtering signals
        ext_embed_B1 = self.ext_attention_encoder_1(item_embed_B, is_A=False)

        # 3 using EA_channel_2 for positional encoding and sequential pattern
        ext_embed_B2 = self.ext_attention_encoder_2(item_pos_B, dim_coefficient=self.dim_coefficient)

        self.seq_embed_B = ext_embed_B1 + ext_embed_B2

        seq_emb_B_output = torch.concat([self.seq_embed_B, self.user_embed], axis=1)
        seq_emb_B_output = torch.dropout(seq_emb_B_output, self.dropout_rate, self.training)

        seq_emb_AB_output = torch.concat([self.seq_embed_A , self.seq_embed_B], axis=1)
        seq_emb_AB_output = torch.concat([seq_emb_AB_output , self.user_embed], axis=1)
        seq_emb_AB_output = torch.dropout(seq_emb_AB_output, self.dropout_rate, self.training)

        return seq_emb_A_output, seq_emb_B_output , seq_emb_AB_output
    
    def ext_attention_encoder_1(self, seq_ebd, is_A):
        if is_A:
            shape_0, shape_1 = seq_ebd.shape[:2]
            meo_ebd_1 = torch.matmul(torch.reshape(seq_ebd, [-1, self.embedding_size]),
                                    self.all_weights['W_att_A']) + self.all_weights['b_att_A']  # [?, 16]
            meo_ebd_1 = nn.functional.relu(meo_ebd_1)  # [?, 16]
            dim_trans_1 = torch.reshape(torch.matmul(meo_ebd_1, self.all_weights['h_att_A']),
                                        [shape_0, shape_1])  # [?, ?]
            dim_trans_1 = torch.exp(dim_trans_1)  # [?, ?]
            mask_index_A = torch.sum(self.len_A, 1)  # [?, ]
            mask_matrix_A = sequence_mask(mask_index_A, max_len=shape_1, dtype=torch.float32)  # [?, ?]
            masked_ebd_A = mask_matrix_A * dim_trans_1  # [?, ?]
            exp_sum_A = torch.sum(masked_ebd_A, 1, keepdims=True)  # [?, 1]
            exp_sum_A = torch.pow(exp_sum_A, torch.tensor(self.beta))  # [?, 1]

            score_A = torch.unsqueeze(torch.div(masked_ebd_A, exp_sum_A), 2)  # [?, ?, 1]

            return torch.sum(score_A * seq_ebd, 1)
        else:
            shape_0, shape_1 = seq_ebd.shape[:2]
            mlp_output_B = torch.matmul(torch.reshape(seq_ebd, [-1, self.embedding_size]),
                                        self.all_weights['W_att_B']) + self.all_weights['b_att_B']
            mlp_output_B = torch.tanh(mlp_output_B)
            d_trans_B = torch.reshape(torch.matmul(mlp_output_B, self.all_weights['h_att_B']), [shape_0, shape_1])
            d_trans_B = torch.exp(d_trans_B)
            mask_index_B = torch.sum(self.len_B, 1)
            mask_mat_B = sequence_mask(mask_index_B, max_len=shape_1, dtype=torch.float32)
            d_trans_B = mask_mat_B * d_trans_B
            exp_sum_B = torch.sum(d_trans_B, 1, keepdims=True)
            exp_sum_B = torch.pow(exp_sum_B, torch.tensor(self.beta))

            score_B = torch.unsqueeze(torch.div(d_trans_B, exp_sum_B), 2)

            return torch.sum(score_B * seq_ebd, 1)

    def ext_attention_encoder_2(self, seq_ebd, dim_coefficient):  # seq_ebd[?, ?, 16] dim_coef = 4
        input_dim_0, input_dim_1 = seq_ebd.shape[:2]  # ?
        dense_layer_1 = self.fc1(seq_ebd)  # [?, ?, 64]
        reshape_1 = torch.reshape(dense_layer_1, shape=(input_dim_0, input_dim_1, self.num_heads,
                                                     self.embedding_size * dim_coefficient // self.num_heads))  # [?, ?, 2, 32]
        reorder_1 = torch.permute(reshape_1, [0, 2, 1, 3])  # [?, 2, ?, 32]
        # a linear layer for key_vectors
        meo_key_vec = self.fc2(reorder_1)  # [?, 2, ?, 64]
        # normalize attention map
        meo_key_vec = torch.softmax(meo_key_vec, axis=2)  # [?, 2, ?, 64]
        # dobule-normalization
        meo_key_vec = meo_key_vec / (
                self.regular_rate_att + torch.sum(meo_key_vec, axis=-1, keepdims=True))  # [?, 2, ?, 64]
        drop_layer_1 = torch.dropout(meo_key_vec, self.keep_prob, train=self.training)  # [?, 2, ?, 64]

        # a linear layer for value_vectors
        meo_value_vec = self.fc3(drop_layer_1)  # [?, 2, ?, 32]
        reorder_2 = torch.permute(meo_value_vec, dims=[0, 2, 1, 3])  # [?, ?, 2, 32]
        reorder_2 = torch.sum(reorder_2, axis=2)  # 将多头注意力合并 [?, ?, 32]
        # x = torch.reshape(reorder_2, [input_dim_0, input_dim_1, self.embedding_size * dim_coefficient])  # [?, ?, 64]
        # x = torch.sum(x, axis=0)  # [?, 64]
        user_ebd = torch.sum(reorder_2, axis=1)  # [?, 32]
        # a linear layer to project original dim
        out_put_ebd = self.fc4(user_ebd)  # [?, 16]
        out_put_ebd = torch.dropout(out_put_ebd, self.keep_prob, self.training)

        return out_put_ebd
    


def sequence_mask(lens, max_len, dtype:torch.float32):
    num = lens.shape[0]
    mask_mat_B = torch.zeros([num, max_len], dtype=dtype).to(args.device)
    for idx, elem in enumerate(lens):
        mask_mat_B[idx, :elem] = 1
    return mask_mat_B
    