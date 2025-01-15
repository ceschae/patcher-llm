from gpt_download import download_and_load_gpt2
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tiktoken
import time
import torch
import torch.nn as nn
import urllib.request
import zipfile

# copied from ch3/demo.py
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

# copied from ch4/demo.py
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # shorcut connection for attention block
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add the original input back

        shortcut = x # shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # adds the original input back
        return x

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # focus on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1) # probas has shape (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # idx_next has shape (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
    return idx

# copeid from ch5/demo.py
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # the original GPT-2 model reused the token embedding weights in the output layer to reduce the total number of parameters (weight tying)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# copying over

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# prints
    #     Label                                               Text
    # 0      ham  Go until jurong point, crazy.. Available only ...
    # 1      ham                      Ok lar... Joking wif u oni...
    # 2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
    # 3      ham  U dun say so early hor... U c already then say...
    # 4      ham  Nah I don't think he goes to usf, he lives aro...
    # ...    ...                                                ...
    # 5567  spam  This is the 2nd time we have tried 2 contact u...
    # 5568   ham               Will ü b going to esplanade fr home?
    # 5569   ham  Pity, * was in mood for that. So...any other s...
    # 5570   ham  The guy did some bitching but I acted like i'd...
    # 5571   ham                         Rofl. Its true to its name

    # [5572 rows x 2 columns]
print(df)

# prints
    # Label
    # ham     4825
    # spam     747
    # Name: count, dtype: int64
print(df["Label"].value_counts())

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] # count instances of "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) # randomly samples "ham" instances to match the number of "spam" instances
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
# prints
    # Label
    # ham     747
    # spam    747
    # Name: count, dtype: int64
print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) # shuffles the data frame
    train_end = int(len(df) * train_frac) # calculates the split indices
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # test size is implied to be 0.2 as the remainder

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
# prints [50256]
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else: 
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
# prints 120
print(train_dataset.max_length)

val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

for input_batch, target_batch in train_loader:
    pass
# prints Input batch dimensions: torch.Size([8, 120])
print("Input batch dimensions:", input_batch.shape)
# prints Label batch dimensions: torch.Size([8])
print("Label batch dimensions:", target_batch.shape)

# prints 130 training batches
print(f"{len(train_loader)} training batches")
# prints 19 validation batches
print(f"{len(val_loader)} validation batches")
# prints 38 test batches
print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-mmedium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"],
)

# prints
    # Every effort moves you forward.

    # The first step is to understand the importance of your work
print(token_ids_to_text(token_ids, tokenizer))

text_2 = "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"],
)
# prints Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award. You will be notified when your prize is available. You can choose to receive a prize by email or by phone.
print(token_ids_to_text(token_ids, tokenizer))

# prints
    # GPTModel(
    # (tok_emb): Embedding(50257, 768)
    # (pos_emb): Embedding(1024, 768)
    # (drop_emb): Dropout(p=0.0, inplace=False)
    # (trf_blocks): Sequential(
    #     (0): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (1): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (2): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (3): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (4): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (5): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (6): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (7): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (8): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (9): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (10): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    #     (11): TransformerBlock(
    #     (att): MultiHeadAttention(
    #         (W_query): Linear(in_features=768, out_features=768, bias=True)
    #         (W_key): Linear(in_features=768, out_features=768, bias=True)
    #         (W_value): Linear(in_features=768, out_features=768, bias=True)
    #         (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #         (dropout): Dropout(p=0.0, inplace=False)
    #     )
    #     (ff): FeedForward(
    #         (layers): Sequential(
    #         (0): Linear(in_features=768, out_features=3072, bias=True)
    #         (1): GELU()
    #         (2): Linear(in_features=3072, out_features=768, bias=True)
    #         )
    #     )
    #     (norm1): LayerNorm()
    #     (norm2): LayerNorm()
    #     (drop_shortcut): Dropout(p=0.0, inplace=False)
    #     )
    # )
    # (final_norm): LayerNorm()
    # (out_head): Linear(in_features=768, out_features=50257, bias=False)
    # )
print(model)

# freeze the model, make all layers untrainable
for param in model.parameters():
    param.requires_grad = False 

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# make the final LayerNorm and last transformer block trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
# prints Inputs: tensor([[5211,  345,  423,  640]])
print("Inputs:", inputs)
# prints Inputs dimensions: torch.Size([1, 4])
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

with torch.no_grad():
    outputs = model(inputs)
# prints
    # Outputs:
    # tensor([[[-1.5854,  0.9904],
    #         [-3.7235,  7.4548],
    #         [-2.2661,  6.6049],
    #         [-3.5983,  3.9902]]])
print("Outputs:\n", outputs)
# prints Outputs dimensions: torch.Size([1, 4, 2])
print("Outputs dimensions:", outputs.shape)
# prints Last output token: tensor([[-3.5983,  3.9902]])
print("Last output token:", outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
# prints Class label: 1
print("Class label:", label.item())

logits = outputs[:, -1, :]
label = torch.argmax(logits)
# prints Class label: 1
print("Class label:", label.item())

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # logits of last output token
            
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

device = torch.device("cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# prints Training accuracy: 46.25%
print(f"Training accuracy: {train_accuracy*100:.2f}%")
# prints Validation accuracy: 45.00%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# prints Test accuracy: 48.75%
print(f"Test accuracy: {test_accuracy*100:.2f}%")

def calc_loss_batch(input_batch, target_batch, mmodel, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches

with torch.no_grad(): # disable gradient tracking for efficiency because we aren't traininng yet
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
# prints Training loss: 3.854
print(f"Training loss: {train_loss:.3f}")
# prints Validation loss: 3.859
print(f"Validation loss: {val_loss:.3f}")
# prints Test loss: 3.844
print(f"Test loss: {test_loss:.3f}")

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train() # sets the model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate the gradients
            optimizer.step() # updates model weights using loss gradients
            examples_seen += input_batch.shape[0] # tracks examples instead of tokens
            global_step += 1

            # optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# prints Training accuracy: 97.21%
print(f"Training accuracy: {train_accuracy*100:.2f}%")
# prints Validation accuracy: 97.32%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# prints Test accuracy: 95.67%
print(f"Test accuracy: {test_accuracy*100:.2f}%")

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)] # truncate sequences if they are too long
    input_ids += [pad_token_id] * (max_length - len(input_ids)) # pad sequences to longest sequence
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # adds batch dimension
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
# prints spam
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
# prints not spam
print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

torch.save(model.state_dict(), "review_classifier.pth")
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)