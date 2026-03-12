import torch
import numpy as np
from torch.distributions.normal import Normal

from layers import StableTransformerXL

class TransformerGaussianPolicy(torch.nn.Module):
    def __init__(self, state_dim, act_dim, n_transformer_layers=4, n_attn_heads=3):
        ''' 
            NOTE - I/P Shape : [seq_len, batch_size, state_dim]
            static_dim: 观察空间的维度
            act_dim: 动作空间的维度
            n_transformer_layers: transformer层数
            n_attn_heads: 注意力头数
        '''
        super(TransformerGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        # 构建transformer模型，输入维度为状态维度，输出维度也是状态维度，层数和注意力头数根据参数设置
        self.transformer = StableTransformerXL(d_input=state_dim, n_layers=n_transformer_layers, 
            n_heads=n_attn_heads, d_head_inner=32, d_ff_inner=64)
        self.memory = None

        # 预测状态的价值
        self.head_state_value = torch.nn.Linear(state_dim, 1)
        # 预测动作的均值，todo 这里的动作是离散值还是连续值
        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)
        # 这里是动作的标准差的log值
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def _distribution(self, trans_state):
        mean = self.tanh(self.head_act_mean(trans_state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1) 

    def forward(self, state, action=None):
        '''
        state: 输入状态，形状为[seq_len, batch_size, state_dim]
        '''
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state['logits'], trans_state['memory']

        policy = self._distribution(trans_state)
        state_value = self.head_sate_value(trans_state)

        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value

    def step(self, state):
        if state.shape[0] == self.state_dim:
            state = state.reshape(1, 1, -1)
        with torch.no_grad():
            trans_state = self.transformer(state, self.memory)
            trans_state, self.memory = trans_state['logits'], trans_state['memory']

            policy = self._distribution(trans_state)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)
            state_value = self.head_sate_value(trans_state)

        return action.numpy(), logp_a.numpy(), state_value.numpy()

if __name__ == '__main__':
    states = torch.randn(20, 5, 8) # seq_size, batch_size, dim - better if dim % 2 == 0
    print("=> Testing Policy")
    policy = TransformerGaussianPolicy(state_dim=states.shape[-1], act_dim=4)
    for i in range(10):
        act = policy(states) # 输入状态，进行预测
        action = act[0].sample()
        print(torch.isnan(action).any(), action.shape)
