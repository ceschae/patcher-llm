import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your    (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]] # step    (x^6)
)

query = inputs[1] # for example, let's calculate the attn scores for x^2
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs): 
    attn_scores_2[i] = torch.dot(x_i, query)
# prints tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
print(attn_scores_2)

# calculating a dot product (i.e. how similar two vectors are)

res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
# prints tensor(0.9544)
print(res)
# prints tensor(0.9544)
print(torch.dot(inputs[0], query))

attn_weights_2_temp = attn_scores_2 / attn_scores_2.sum()
# prints Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
print("Attention weights:", attn_weights_2_temp)
# prints Sum: tensor(1.0000)
print("Sum:", attn_weights_2_temp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
# prints Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Attention weights:", attn_weights_2_naive)
# prints Sum: tensor(1.)
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# prints Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Attention weights:", attn_weights_2)
# prints Sum: tensor(1.)
print("Sum:", attn_weights_2.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
# prints tensor([0.4419, 0.6515, 0.5683])
print(context_vec_2)

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
# prints
    # tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
    #         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
    #         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
    #         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
    #         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
    #         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
print(attn_scores)

attn_scores = inputs @ inputs.T # matrix multiplication syntax
# prints
    # tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
    #         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
    #         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
    #         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
    #         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
    #         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
# prints
    # tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
    #         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
    #         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
    #         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
    #         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
    #         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# prints Row 2 sum: 1.0
print("Row 2 sum:", row_2_sum)
# prints All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
# prints
    # tensor([[0.4421, 0.5931, 0.5790],
    #         [0.4419, 0.6515, 0.5683],
    #         [0.4431, 0.6496, 0.5671],
    #         [0.4304, 0.6298, 0.5510],
    #         [0.4671, 0.5910, 0.5266],
    #         [0.4177, 0.6503, 0.5645]])
print(all_context_vecs)
# prints Previous 2nd context vector: tensor([0.4419, 0.6515, 0.5683])
print("Previous 2nd context vector:", context_vec_2)

# implementing self-attention with trainable weights

x_2 = inputs[1] # the second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# prints tensor([0.4306, 1.4551])
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
# prints keys.shape: torch.Size([6, 2])
print("keys.shape:", keys.shape)
# prints values.shape: torch.Size([6, 2])
print("values.shape:", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# prints tensor(1.8524)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T # all attention scores for given query
# prints tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# prints tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
# print tensor([0.3061, 0.8210])
print(context_vec_2)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# prints
    # tensor([[0.2996, 0.8053],
    #         [0.3061, 0.8210],
    #         [0.3058, 0.8203],
    #         [0.2948, 0.7939],
    #         [0.2927, 0.7891],
    #         [0.2990, 0.8040]], grad_fn=<MmBackward0>)
print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
# prints
    # tensor([[-0.5337, -0.1051],
    #         [-0.5323, -0.1080],
    #         [-0.5323, -0.1079],
    #         [-0.5297, -0.1076],
    #         [-0.5311, -0.1066],
    #         [-0.5299, -0.1081]], grad_fn=<MmBackward0>)
print(sa_v2(inputs)) 

# hiding future words with casual attention

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# prints
    # tensor([[0.1717, 0.1762, 0.1761, 0.1555, 0.1627, 0.1579],
    #         [0.1636, 0.1749, 0.1746, 0.1612, 0.1605, 0.1652],
    #         [0.1637, 0.1749, 0.1746, 0.1611, 0.1606, 0.1651],
    #         [0.1636, 0.1704, 0.1702, 0.1652, 0.1632, 0.1674],
    #         [0.1667, 0.1722, 0.1721, 0.1618, 0.1633, 0.1639],
    #         [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],
    #     grad_fn=<SoftmaxBackward0>)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# prints
    # tensor([[1., 0., 0., 0., 0., 0.],
    #         [1., 1., 0., 0., 0., 0.],
    #         [1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 0.],
    #         [1., 1., 1., 1., 1., 1.]])
print(mask_simple)

masked_simple = attn_weights*mask_simple
# prints
    # tensor([[0.1717, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.1636, 0.1749, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.1637, 0.1749, 0.1746, 0.0000, 0.0000, 0.0000],
    #         [0.1636, 0.1704, 0.1702, 0.1652, 0.0000, 0.0000],
    #         [0.1667, 0.1722, 0.1721, 0.1618, 0.1633, 0.0000],
    #         [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],
    #     grad_fn=<MulBackward0>)
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
# prints
    # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
    #         [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
    #         [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
    #         [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],
    #     grad_fn=<DivBackward0>)
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# prints
    # tensor([[0.3111,   -inf,   -inf,   -inf,   -inf,   -inf],
    #         [0.1655, 0.2602,   -inf,   -inf,   -inf,   -inf],
    #         [0.1667, 0.2602, 0.2577,   -inf,   -inf,   -inf],
    #         [0.0510, 0.1080, 0.1064, 0.0643,   -inf,   -inf],
    #         [0.1415, 0.1875, 0.1863, 0.0987, 0.1121,   -inf],
    #         [0.0476, 0.1192, 0.1171, 0.0731, 0.0477, 0.0966]],
    #     grad_fn=<MaskedFillBackward0>)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# prints
    # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
    #         [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
    #         [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
    #         [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],
    #     grad_fn=<SoftmaxBackward0>)
print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
# prints
    # tensor([[2., 2., 0., 2., 2., 0.],
    #         [0., 0., 0., 2., 0., 2.],
    #         [2., 2., 2., 2., 0., 2.],
    #         [0., 2., 2., 0., 0., 2.],
    #         [0., 2., 0., 2., 0., 2.],
    #         [0., 2., 2., 2., 2., 0.]])
print(dropout(example))

torch.manual_seed(123)
# prints
    # tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #         [0.6380, 0.6816, 0.6804, 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.5090, 0.5085, 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.4120, 0.0000, 0.3869, 0.0000, 0.0000],
    #         [0.0000, 0.3418, 0.3413, 0.3308, 0.3249, 0.0000]],
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
# prints torch.Size([2, 6, 3])
print(batch.shape)

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # transpose dimensions 1 and to, keeping the batch dimension at the first position (0)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print context_vecs.shape: torch.Size([2, 6, 2])
print("context_vecs.shape:", context_vecs.shape)

# extending single-head attention to multi-head attention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1] # this is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# prints
    # tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
    #         [-0.5874,  0.0058,  0.5891,  0.3257],
    #         [-0.6300, -0.0632,  0.6202,  0.3860],
    #         [-0.5675, -0.0843,  0.5478,  0.3589],
    #         [-0.5526, -0.0981,  0.5321,  0.3428],
    #         [-0.5299, -0.1081,  0.5077,  0.3493]],

    #         [[-0.4519,  0.2216,  0.4772,  0.1063],
    #         [-0.5874,  0.0058,  0.5891,  0.3257],
    #         [-0.6300, -0.0632,  0.6202,  0.3860],
    #         [-0.5675, -0.0843,  0.5478,  0.3589],
    #         [-0.5526, -0.0981,  0.5321,  0.3428],
    #         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
print(context_vecs)
# prints context_vecs.shape: torch.Size([2, 6, 4])
print("context_vecs.shape:", context_vecs.shape)

# to solve exercise 3.2, change d_out to 1

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduces the projection dimension to match the desired output dimension
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicitly split the matrix by adding a num_heads dimension.
        # then unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose the shape to match dimensions
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # computes dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        # masks truncated to number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        # combines heads (self.d_out = self.num_heads * self.head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # adds optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec

# tensor shape is b=1, num_heads=2, num_tokens=3, head_dim=4
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                    
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
# prints
    # tensor([[[[1.3208, 1.1631, 1.2879],
    #         [1.1631, 2.2150, 1.8424],
    #         [1.2879, 1.8424, 2.0402]],

    #         [[0.4391, 0.7003, 0.5903],
    #         [0.7003, 1.3737, 1.0620],
    #         [0.5903, 1.0620, 0.9912]]]])
print(a @ a.transpose(2, 3))

first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
# prints
    # first head:
    # tensor([[1.3208, 1.1631, 1.2879],
    #         [1.1631, 2.2150, 1.8424],
    #         [1.2879, 1.8424, 2.0402]])
print("first head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
# prints
    # second head:
    # tensor([[0.4391, 0.7003, 0.5903],
    #         [0.7003, 1.3737, 1.0620],
    #         [0.5903, 1.0620, 0.9912]])
print("\nsecond head:\n", second_res)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# prints
    # tensor([[[0.3190, 0.4858],
    #         [0.2943, 0.3897],
    #         [0.2856, 0.3593],
    #         [0.2693, 0.3873],
    #         [0.2639, 0.3928],
    #         [0.2575, 0.4028]],

    #         [[0.3190, 0.4858],
    #         [0.2943, 0.3897],
    #         [0.2856, 0.3593],
    #         [0.2693, 0.3873],
    #         [0.2639, 0.3928],
    #         [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
print(context_vecs)
# prints context_vecs.shape: torch.Size([2, 6, 2])
print("context_vecs.shape:", context_vecs.shape)

# to solve exercise 3.3, set num_heads=12, context_length=1024, d_in=d_out=768