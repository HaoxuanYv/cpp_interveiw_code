
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mast = None):
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)

        attn = torch.softmax(scores, dim = -1)

        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.fc_out(context)
        return output
