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
        # 存储每一层输入的状态，即为记忆，初始值为None，后续会在forward函数中更新
        self.memory = None

        # 预测状态的价值
        self.head_state_value = torch.nn.Linear(state_dim, 1)
        # 预测动作的均值，所以是连续动作
        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)
        # 这里是动作的标准差的log值
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def _distribution(self, trans_state):
        # 这里构建的是一个高斯分布的动作采样，所以当前的动作是一个连续的动作
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
        trans_state, self.memory = trans_state['logits'], trans_state['memory'] # 得到transformer的输出状态和更新后的记忆，输出状态的形状为[seq_len, batch_size, state_dim]，记忆的形状为[seq_len, batch_size, state_dim]，这里的记忆是为了在下一次前向传播中使用，供transformer计算注意力分数使用

        policy = self._distribution(trans_state) # 根据输出的状态获取动作的分布
        state_value = self.head_sate_value(trans_state) # 预测价值

        logp_a = None
        if action is not None:
            # 如果有传入动作则直接选中动作对应的概率，否则直接返回动作分布的log概率
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value # 返回动作分布，动作的log概率，状态的价值

    def step(self, state):
        '''
        这里就是用于采样时的函数，输入状态，输出动作，动作的log概率，状态的价值，动作时随机采样的，动作的log概率是根据动作分布计算出来的，状态的价值是根据输出状态预测出来的
        从这里来看，这是一个针对连续动作空间的策略，如果是离散动作空间的话，动作分布应该是Categorical而不是Normal，动作的采样也应该是从Categorical分布中采样，而不是从Normal分布中采样，动作的log概率也应该是Categorical分布的log_prob而不是Normal分布的log_prob，这些都是根据具体的任务和环境来决定的，这里假设是连续动作空间，所以使用了Normal分布来建模动作分布，使用了tanh函数来限制动作的范围在[-1, 1]之间，这样可以更好地适应一些连续控制任务中的动作范围要求
        '''
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
