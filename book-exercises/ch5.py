from ch2 import create_dataloader_v1
from ch4 import GPTModel

from gpt_download import download_and_load_gpt2
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import urllib.request

# idx is a (batch, n_tokens) array of indices in the current context
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
# copying over

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

# prints
    # Output text:
    # Every effort moves you rentingetic wasnم refres RexMeCHicular stren
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40,    1107, 588]])  #  "I really like"]

targets = torch.tensor([[3626, 6100, 345],   # [" effort moves you",
                        [1107, 588, 11311]]) #  " really like chocolate"]

with torch.no_grad(): # disables gradient tracking since we're not training yet
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # probability of each token in vocabulary
# prints torch.Size([2, 3, 50257])
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# prints
    # Token IDs:
    # tensor([[[16657], # first batch
    #         [  339],
    #         [42826]],

    #         [[49906], # second batch
    #         [29669],
    #         [41751]]])
print("Token IDs:\n", token_ids)

# prints Targets batch 1:  effort moves you
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# prints Outputs batch 1:  Armed heNetflix
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# prints Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# prints Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# prints tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])
print(log_probas)

avg_log_probas = torch.mean(log_probas)
# prints tensor(-10.7940)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
# prints tensor(10.7940)
print(neg_avg_log_probas)

# prints Logits shape: torch.Size([2, 3, 50257])
print("Logits shape:", logits.shape)
# prints Targets shape: torch.Size([2, 3])
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# prints Flattened logits: torch.Size([6, 50257])
print("Flattened logits:", logits_flat.shape)
# prints Flattened targets: torch.Size([6])
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# prints tensor(10.7940)
print(loss)

file_path = "the-verdict.txt"
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# prints Characters: 20479
print("Characters:", total_characters)
# prints Tokens: 5145
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# prints
    # Train loader:
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
    # torch.Size([2, 256]) torch.Size([2, 256])
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

# prints
    # Validation loader:
    # torch.Size([2, 256]) torch.Size([2, 256])
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # iterate over all batches if no num_batches is specified
    else:
        num_batches = min(num_batches, len(data_loader)) # reduces num of batches to total num of batches if num_batches > number of batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # averages loss over all the batches

device = torch.device("cpu")
model.to(device)
with torch.no_grad(): # disable gradient tracking for efficiency since we're not training yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

# prints Training loss: 10.987583372328016
print("Training loss:", train_loss)
# prints Validation loss: 10.98110580444336
print("Validation loss:", val_loss)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients
            optimizer.step() # update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0: # optional evaluation step
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # dropout is disabled during evaluation for stable, reproducable results
    with torch.no_grad(): # disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # parameters() returns all trainable weight parameters of the model
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() # create a second x axis that shares the same y axis
    ax2.plot(tokens_seen, train_losses, alpha=0) # invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

model.to("cpu")
model.train()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"])
# prints
    # Output text:
    # Every effort moves you know," was one of the axioms he laid down across the last word.
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6, 
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# prints forward
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# prints forward
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# prints
    # 73 x closer
    # 0 x every
    # 0 x effort
    # 582 x forward
    # 2 x inches
    # 0 x moves
    # 0 x pizza
    # 343 x toward
print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5] # original, lower, and higher confidence
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# prints Top logits: tensor([6.7500, 6.2800, 4.5100])
print("Top logits:", top_logits)
# prints Top positions: tensor([3, 7, 0])
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1], # identifies logits less than the minimum in the top 3
    input=torch.tensor(float('-inf')), # assigns -inf to lower logits
    other=next_token_logits # retains the original logits for all other tokens
)
# prints tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
# prints tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
print(topk_probas)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0: 
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break # stops generating early if end-of-sequence token is encountered
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
        
torch.manual_seed(123)
token_ids = generate(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4)
# prints
    # Output text:
    # Every effort moves youlit terrace.



    # " he said deprecating laugh
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

torch.save(model.state_dict(), "ch5-model.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("ch5-model.pth", map_location=device))
model.eval()

torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),}, "ch5-model_and_optimizer.pth")

checkpoint = torch.load("ch5-model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();

urllib.request.urlretrieve("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py", "gpt_download.py")

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# prints Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
print("Settings:", settings)
# prints Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
print("Parameter dictionary keys:", params.keys())

# prints
    # Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
    # [[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
    # 0.04531523]
    # [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
    # 0.04318958]
    # [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
    # -0.08785918]
    # ...
    # [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
    # -0.06952604]
    # [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
    # -0.02245961]
    # [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
    # 0.12067825]]
print(params["wte"])
# prints Token embedding weight tensor dimensions: (50257, 768)
print("Token embedding weight tensor dimensions:", params["wte"].shape)

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (772M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

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

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt, 
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, 
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

# prints
    # output text:
    # Every effort moves you toward finding an ideal new way to practice something!

    # What makes us want to be on top of that?
print("output text:\n", token_ids_to_text(token_ids, tokenizer))