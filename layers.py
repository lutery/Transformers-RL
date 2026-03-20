import torch
import torch.nn.functional as F


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        '''
        dim: positional embedding的维度，通常和输入的维度相同
        '''
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        # 这边是位置频率，用于构建位置编码中，每一个位置维度的频率，论文中建议设置为10000
        # inv_freq shape is (dim/2)，每个位置维度的频率，频率越高的位置维度变化越快，频率越低的位置维度变化越慢
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        '''
        positions: 位置索引，形状为[seq_len]，表示每个位置的索引，当前序列的位置索引为0，历史状态的位置索引为正数，距离越远的历史状态位置索引越大
        '''
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq) # 看markdown
        # 然后对计算得到的频率分别进行sin和cos变换，得到位置编码的不同维度，最后将sin和cos的结果拼接起来，得到最终的位置编码，形状为[seq_len, dim]
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :] # 在位置编码的基础上增加一个维度，形状变为[seq_len, 1, dim]，这个维度是为了后续计算注意力时能够广播到所有的批次


class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        '''
        d_input: 输入的维度，目前也是状态的维度
        d_inner: feed forward层的维度
        dropout: dropout概率

        todo 这个的作用是什么，为什么要在transformer中使用这个位置前馈网络
        '''
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        '''
        d_input: 输入的维度，目前也是状态的维度
        bg: gate的偏置项，论文中建议设置为0.1 todo
        '''
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        '''
        d_input: 输入的维度，目前也是状态的维度
        d_inner: 每个头的维度
        n_heads: 注意力头数
        dropout: dropout概率
        dropouta: 注意力dropout概率
        '''
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)

        # for positional embeddings.
        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(d_inner * n_heads, d_input, bias=False)
        self.dropo = torch.nn.Dropout(dropout)

    def _rel_shift(self, x):
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        return (
            torch.cat([zero_pad, x], dim=1)
            .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
            .view_as(x)
        )

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8) 输入的当前状态序列，seq是当前序列的长度，bs是批次大小，d_input是输入的维度
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8) 位置编码，包含当前序列和历史状态序列的位置编码，seq是当前序列的长度，prev_seq是历史状态序列的长度，bs是批次大小，d_input是输入的维度
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8) 历史上的每层的状态序列，prev_seq是历史状态序列的长度，b是批次大小，d_in是输入的维度
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x ) 全局参数，用于内容注意力的计算，num_heads是注意力头数，inner_dim是每个头的维度
            - v: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x ) 全局参数，用于位置注意力的计算，num_heads是注意力头数，inner_dim是每个头的维度
            - mask: torch.FloatTensor, Optional = (20, 40, 1) 注意力掩码，用于屏蔽掉当前序列中不可见的位置，通常是当前序列中未来的位置和历史状态中不可见的位置，shape为[seq, seq + prev_seq, 1]，seq是当前序列的长度，prev_seq是历史状态序列的长度

        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)

        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner # H 注意力头数、d 每个头的维度
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        # 将历史状态序列和当前状态序列在序列维度上拼接起来，得到一个新的输入序列，这个序列包含了当前状态和历史状态的信息，供后续的注意力计算使用
        input_with_memory = torch.cat([memory, input_], dim=0)

        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        # todo 这里为啥kv一起计算，q单独计算
        # 将拼接后的输入序列通过线性变换得到键和值的表示
        k_tfmd, v_tfmd = torch.chunk(
            self.linear_kv(input_with_memory),
            2,
            dim=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        # 进一步提取特征得到q
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1] # q k v的批次大小应该相同

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = torch.softmax(attn * self.scale, dim=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(
        self,
        n_heads,
        d_input,
        d_head_inner,
        d_ff_inner,
        dropout,
        gating=True,
        dropouta=0.0,
    ):
        '''
        n_heads: 注意力头数
        d_input: 输入的维度，目前也是状态的维度
        d_head_inner: 每个头的维度
        d_ff_inner: feed forward层的维度
        dropout: dropout概率
        gating: 是否使用门控机制
        dropouta: 注意力dropout概率
        '''
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        # 以下两个门的作用是什么
        # 据说这两个门是用来替代残差链接的，确认有多少历史的信息要记录、有多少现在的信息要保留、有多少新的信息要引入，来更好地控制信息流动，防止梯度消失或者爆炸。
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        # 多头注意力层
        self.mha = MultiHeadAttentionXL(
            d_input,
            d_head_inner,
            n_heads=n_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        # todo 这个是干嘛的？
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = torch.nn.LayerNorm(d_input)
        self.norm2 = torch.nn.LayerNorm(d_input)

    def forward(self, input_, pos_embs, u, v, mask=None, mems=None):
        '''
        input_: 输入的当前最新的状态记忆
        pos_embes: 位置编码，包含当前状态和历史状态的位置编码
        u, v: 全局参数，用于多头注意力层中的内容和位置注意力的计算 todo
        mask: 注意力掩码，用于屏蔽掉当前序列中不可见的位置，通常是当前序列中未来的位置和历史状态中不可见的位置
        mems: 历史状态序列，用于多头注意力层中的键和值的计算
        '''

        src2 = self.norm1(input_) # 对输入的当前状态进行层归一化，得到src2
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)
        src = self.gate1(input_, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src


class StableTransformerXL(torch.nn.Module):
    def __init__(
        self,
        d_input,
        n_layers,
        n_heads,
        d_head_inner,
        d_ff_inner,
        dropout=0.1,
        dropouta=0.0,
        mem_len=100,
    ):
        '''
        d_input: 输入的维度，目前也是状态的维度
        n_layers: transformer层数
        n_heads: 注意力头数
        d_head_inner: 每个头的维度
        d_ff_inner: feed forward层的维度
        dropout: dropout概率
        dropouta: 注意力dropout概率
        mem_len: memory长度
        '''
        super(StableTransformerXL, self).__init__()

        (
            self.n_layers,
            self.n_heads,
            self.d_input,
            self.d_head_inner,
            self.d_ff_inner,
        ) = (n_layers, n_heads, d_input, d_head_inner, d_ff_inner)

        self.pos_embs = PositionalEmbedding(d_input) # 构建位置编码器
        self.drop = torch.nn.Dropout(dropout) # 输入dropout
        self.mem_len = mem_len
        # 构建编码器层，使用ModuleList来存储多个编码器层
        self.layers = torch.nn.ModuleList(
            [
                StableTransformerEncoderLayerXL(
                    n_heads,
                    d_input,
                    d_head_inner=d_head_inner,
                    d_ff_inner=d_ff_inner,
                    dropout=dropout,
                    dropouta=dropouta,
                )
                for _ in range(n_layers)
            ]
        )

        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
        )

    def init_memory(self, device=torch.device("cpu")):
        """
        初始化历史状态序列

        + Arguments
            - device: torch.device, 指定设备
        + Returns
            - List[torch.FloatTensor], 每一层的历史状态序列，初始为空
        """
        return [
            torch.empty(0, dtype=torch.float).to(device)
            for _ in range(self.n_layers + 1)
        ]

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None):
        """
        + Arguments
            - inputs - torch.FloatTensor = [T x B x d_inner] = [20 x 5 x 8] 输入状态序列 shape 为[seq_len, batch_size, state_dim]
            - memory - Optional, list[torch.FloatTensor] = [[T x B x d_inner] x 5] 历史状态序列，每一层的历史状态 shape 为[prev_seq_len, batch_size, state_dim]
        """
        if memory is None: 
            # 如果没有提供历史状态序列，就初始化一个空的历史状态序列
            # 返回的memory是一个列表，包含n_layers + 1个元素，每个元素都是一个形状为[0]的空张量
            memory = self.init_memory(inputs.device)
        # 我记得记忆必须是每一层的历史状态序列，所以长度应该是n_layers + 1，确认一下
        assert len(memory) == len(self.layers) + 1 

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0) # prev_seq 这个shape是什么？是历史状态序列的长度吗？确认一下

        # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
        # torch.ones 实在构建一个shape 为(curr_seq, cur_seq + prev_seq)的全1张量，当前序列的每个位置，对“历史 + 当前序列”所有位置的注意力关系。
        # torch.triu 会保留矩阵的上三角部分，其余置 0。diagonal=1 + prev_seq 表示从第1 + prev_seq条对角线开始保留上三角部分，这部分是历史 memory，这部分应该全部可见
        # bool() 表示将0/1张量转换为bool类型， 0表示可见，1表示要屏蔽，所以上三角矩阵中上三角的部分就是要屏蔽的部分
        # [..., None] 表示在最后一个维度上增加一个维度，使得mask的形状从[cur_seq, cur_seq + prev_seq]变成[cur_seq, cur_seq + prev_seq, 1]，这样在后续计算注意力时可以广播到所有的头和批次
        dec_attn_mask = (
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            )
            .bool()[..., None]
            .to(inputs.device)
        )

        # 位置编码，看起来应该是仅针对历史状态的位置编码
        # pos_ips shape is （cur_seq + prev_seq），是一个从(cur_seq + prev_seq - 1)到0的递减序列，表示每个位置相对于当前序列的距离，当前序列的位置是0，历史状态的位置是正数，距离越远的历史状态位置数值越大
        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(
            inputs.device
        )
        # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
        # pos_embs shape is (cur_seq + prev_seq, 1, d_input)，每个位置的编码，位置编码的维度和输入的维度相同
        pos_embs = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        # hidden_states 是一个列表，包含每一层的输入状态
        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
            layer_out = layer(
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=dec_attn_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]
        memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": memory}
