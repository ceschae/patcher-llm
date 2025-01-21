from ch4 import GPTModel
from ch5 import calc_loss_loader, generate, load_weights_into_gpt, plot_losses, text_to_token_ids, token_ids_to_text, train_model_simple
from functools import partial
from gpt_download import download_and_load_gpt2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import json
import os
import psutil
import re
import tiktoken
import time
import torch
import urllib.request

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)
# prints Number of entries: 1100
print("Number of entries:", len(data))
# prints 
    # Example entry:
    # {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}
print("Example entry:\n", data[50])
# prints
    # Another example entry:
    # {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
print("Another example entry:\n", data[999])

def format_input(entry):
    instruction_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}"
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# prints
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # Identify the correct spelling of the following word.

    # ### Input:
    # Ocassion

    # ### Response:
    # The correct spelling is 'Occasion.'
print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
# prints
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # What is an antonym of 'complicated'?

    # ### Response:
    # An antonym of 'complicated' is 'simple'.
print(model_input + desired_response)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# prints Training set length: 935
print("Training set length:", len(train_data))
# prints Validation set length: 55
print("Validation set length:", len(val_data))
# prints Test set length: 110
print("Test set length:", len(test_data))

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

tokenizer = tiktoken.get_encoding("gpt2")
# prints [50256]
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (inputs_1, inputs_2, inputs_3)
# prints
    # tensor([[    0,     1,     2,     3,     4],
    #         [    5,     6, 50256, 50256, 50256],
    #         [    7,     8,     9, 50256, 50256]])
print(custom_collate_draft_1(batch))

def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) # shifts +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
# prints
    # tensor([[    0,     1,     2,     3,     4],
    #         [    5,     6, 50256, 50256, 50256],
    #         [    7,     8,     9, 50256, 50256]])
print(inputs)
# prints
    # tensor([[    1,     2,     3,     4, 50256],
    #         [    6, 50256, 50256, 50256, 50256],
    #         [    8,     9, 50256, 50256, 50256]])
print(targets)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
# prints
    # tensor([[    0,     1,     2,     3,     4],
    #         [    5,     6, 50256, 50256, 50256],
    #         [    7,     8,     9, 50256, 50256]])
print(inputs)
# prints
    # tensor([[    1,     2,     3,     4, 50256],
    #         [    6, 50256,  -100,  -100,  -100],
    #         [    8,     9, 50256,  -100,  -100]])
print(targets)

logits_1 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
targets_1 = torch.tensor([0, 1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
# prints tensor(1.1269)
print(loss_1)

logits_2 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
# prints tensor(0.7936)
print(loss_2)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
# prints tensor(1.1269)
print(loss_3)
# prints loss 1 == loss 3: tensor(False)
print("loss 1 == loss 3:", loss_1 == loss_3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
# prints Device: mps
print("Device:", device)

# editor's note: mps was significantly slower for me than cpu, so i'm deviating from the book in this regard
device = torch.device("cpu")
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# prints Train loader:
print("Train loader:")
# prints
    # torch.Size([8, 61]) torch.Size([8, 61])
    # torch.Size([8, 76]) torch.Size([8, 76])
    # torch.Size([8, 73]) torch.Size([8, 73])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 72]) torch.Size([8, 72])
    # torch.Size([8, 80]) torch.Size([8, 80])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 62]) torch.Size([8, 62])
    # torch.Size([8, 75]) torch.Size([8, 75])
    # torch.Size([8, 62]) torch.Size([8, 62])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 77]) torch.Size([8, 77])
    # torch.Size([8, 69]) torch.Size([8, 69])
    # torch.Size([8, 79]) torch.Size([8, 79])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 80]) torch.Size([8, 80])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 69]) torch.Size([8, 69])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 60]) torch.Size([8, 60])
    # torch.Size([8, 59]) torch.Size([8, 59])
    # torch.Size([8, 69]) torch.Size([8, 69])
    # torch.Size([8, 63]) torch.Size([8, 63])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 76]) torch.Size([8, 76])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 91]) torch.Size([8, 91])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 75]) torch.Size([8, 75])
    # torch.Size([8, 89]) torch.Size([8, 89])
    # torch.Size([8, 59]) torch.Size([8, 59])
    # torch.Size([8, 88]) torch.Size([8, 88])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 70]) torch.Size([8, 70])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 74]) torch.Size([8, 74])
    # torch.Size([8, 76]) torch.Size([8, 76])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 75]) torch.Size([8, 75])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 69]) torch.Size([8, 69])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 60]) torch.Size([8, 60])
    # torch.Size([8, 60]) torch.Size([8, 60])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 80]) torch.Size([8, 80])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 61]) torch.Size([8, 61])
    # torch.Size([8, 58]) torch.Size([8, 58])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 63]) torch.Size([8, 63])
    # torch.Size([8, 87]) torch.Size([8, 87])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 71]) torch.Size([8, 71])
    # torch.Size([8, 61]) torch.Size([8, 61])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 65]) torch.Size([8, 65])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 60]) torch.Size([8, 60])
    # torch.Size([8, 72]) torch.Size([8, 72])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 70]) torch.Size([8, 70])
    # torch.Size([8, 57]) torch.Size([8, 57])
    # torch.Size([8, 72]) torch.Size([8, 72])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 62]) torch.Size([8, 62])
    # torch.Size([8, 74]) torch.Size([8, 74])
    # torch.Size([8, 80]) torch.Size([8, 80])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 70]) torch.Size([8, 70])
    # torch.Size([8, 91]) torch.Size([8, 91])
    # torch.Size([8, 61]) torch.Size([8, 61])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 80]) torch.Size([8, 80])
    # torch.Size([8, 81]) torch.Size([8, 81])
    # torch.Size([8, 74]) torch.Size([8, 74])
    # torch.Size([8, 82]) torch.Size([8, 82])
    # torch.Size([8, 63]) torch.Size([8, 63])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 68]) torch.Size([8, 68])
    # torch.Size([8, 67]) torch.Size([8, 67])
    # torch.Size([8, 77]) torch.Size([8, 77])
    # torch.Size([8, 91]) torch.Size([8, 91])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 61]) torch.Size([8, 61])
    # torch.Size([8, 75]) torch.Size([8, 75])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 78]) torch.Size([8, 78])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 64]) torch.Size([8, 64])
    # torch.Size([8, 83]) torch.Size([8, 83])
    # torch.Size([8, 66]) torch.Size([8, 66])
    # torch.Size([8, 74]) torch.Size([8, 74])
    # torch.Size([8, 69]) torch.Size([8, 69])
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
# prints
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # Convert the active sentence to passive: 'The chef cooks the meal every day.'
print(input_text)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=502566,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
# prints
    # ### Response:

    # The chef cooks the meal every day.

    # ### Instruction:

    # Convert the active sentence to passive: 'The chef cooks the
print(response_text)

model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

# prints Training loss: 3.825893831253052
print("Training loss:", train_loss)
# prints Validation loss: 3.7619182586669924
print("Validation loss:", val_loss)

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

# prints
    # Ep 1 (Step 000000): Train loss 2.637, Val loss 2.626
    # Ep 1 (Step 000005): Train loss 1.174, Val loss 1.103
    # Ep 1 (Step 000010): Train loss 0.872, Val loss 0.945
    # Ep 1 (Step 000015): Train loss 0.857, Val loss 0.906
    # Ep 1 (Step 000020): Train loss 0.776, Val loss 0.881
    # Ep 1 (Step 000025): Train loss 0.754, Val loss 0.859
    # Ep 1 (Step 000030): Train loss 0.799, Val loss 0.836
    # Ep 1 (Step 000035): Train loss 0.714, Val loss 0.808
    # Ep 1 (Step 000040): Train loss 0.672, Val loss 0.806
    # Ep 1 (Step 000045): Train loss 0.633, Val loss 0.790
    # Ep 1 (Step 000050): Train loss 0.662, Val loss 0.783
    # Ep 1 (Step 000055): Train loss 0.760, Val loss 0.764
    # Ep 1 (Step 000060): Train loss 0.719, Val loss 0.743
    # Ep 1 (Step 000065): Train loss 0.652, Val loss 0.735
    # Ep 1 (Step 000070): Train loss 0.532, Val loss 0.729
    # Ep 1 (Step 000075): Train loss 0.569, Val loss 0.729
    # Ep 1 (Step 000080): Train loss 0.605, Val loss 0.725
    # Ep 1 (Step 000085): Train loss 0.509, Val loss 0.709
    # Ep 1 (Step 000090): Train loss 0.562, Val loss 0.691
    # Ep 1 (Step 000095): Train loss 0.500, Val loss 0.681
    # Ep 1 (Step 000100): Train loss 0.502, Val loss 0.677
    # Ep 1 (Step 000105): Train loss 0.564, Val loss 0.670
    # Ep 1 (Step 000110): Train loss 0.555, Val loss 0.667
    # Ep 1 (Step 000115): Train loss 0.508, Val loss 0.664
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is prepared every day by the chef.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive:
    # Ep 2 (Step 000120): Train loss 0.435, Val loss 0.672
    # Ep 2 (Step 000125): Train loss 0.451, Val loss 0.686
    # Ep 2 (Step 000130): Train loss 0.447, Val loss 0.682
    # Ep 2 (Step 000135): Train loss 0.405, Val loss 0.682
    # Ep 2 (Step 000140): Train loss 0.410, Val loss 0.681
    # Ep 2 (Step 000145): Train loss 0.369, Val loss 0.681
    # Ep 2 (Step 000150): Train loss 0.381, Val loss 0.676
    # Ep 2 (Step 000155): Train loss 0.412, Val loss 0.676
    # Ep 2 (Step 000160): Train loss 0.415, Val loss 0.684
    # Ep 2 (Step 000165): Train loss 0.379, Val loss 0.686
    # Ep 2 (Step 000170): Train loss 0.323, Val loss 0.682
    # Ep 2 (Step 000175): Train loss 0.337, Val loss 0.670
    # Ep 2 (Step 000180): Train loss 0.393, Val loss 0.658
    # Ep 2 (Step 000185): Train loss 0.416, Val loss 0.659
    # Ep 2 (Step 000190): Train loss 0.340, Val loss 0.650
    # Ep 2 (Step 000195): Train loss 0.330, Val loss 0.637
    # Ep 2 (Step 000200): Train loss 0.310, Val loss 0.637
    # Ep 2 (Step 000205): Train loss 0.352, Val loss 0.632
    # Ep 2 (Step 000210): Train loss 0.367, Val loss 0.631
    # Ep 2 (Step 000215): Train loss 0.396, Val loss 0.635
    # Ep 2 (Step 000220): Train loss 0.301, Val loss 0.649
    # Ep 2 (Step 000225): Train loss 0.349, Val loss 0.662
    # Ep 2 (Step 000230): Train loss 0.294, Val loss 0.658
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is cooked every day by the chef.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: What is the capital of the United Kingdom
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
# prints Training completed in 171.25 minutes
print(f"Training completed in {execution_time_minutes:.2f} minutes")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.manual_seed(123)
# prints
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # Rewrite the sentence using a simile.

    # ### Input:
    # The car is very fast.

    # Correct response:
    # >> The car is as fast as lightning.

    # Model response:
    # >> The car is as fast as a cheetah.
    # -------------------------------------
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # What type of cloud is typically associated with thunderstorms?

    # Correct response:
    # >> The type of cloud typically associated with thunderstorms is cumulonimbus.

    # Model response:
    # >> The type of cloud associated with thunderstorms is a cumulus cloud.
    # -------------------------------------
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # Name the author of 'Pride and Prejudice'.

    # Correct response:
    # >> Jane Austen.

    # Model response:
    # >> The author of 'Pride and Prejudice' is Jane Austen.
    # -------------------------------------
for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}\n\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

# prints {'instruction': 'Rewrite the sentence using a simile.', 'input': 'The car is very fast.', 'output': 'The car is as fast as lightning.', 'model_response': 'The car is as fast as a cheetah.'}
print(test_data[0])

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)
# prints Model saved as gpt2-medium355M-sft.pth
print(f"Model saved as {file_name}")

# took an ollama gander
# prints
    # >>> What do llamas eat?
    # Llamas are ruminant animals, which means they have a four-chambered 
    # stomach and primarily feed on plant-based foods. Their diet typically 
    # consists of:

    # 1. Grasses: Llamas love to graze on grasses, including tall grasses, hay, 
    # and pasture.
    # 2. Leaves: They enjoy munching on leaves from trees and shrubs, such as 
    # willow, alder, and oak.
    # 3. Fruits and vegetables: Llamas can be fed fruits like apples, carrots, 
    # and sweet potatoes, as well as leafy greens like kale and spinach.
    # 4. Hay: A high-quality hay, such as timothy or alfalfa, is a staple in a 
    # llama's diet.
    # 5. Grains: Whole grains like oats, barley, and corn can be used as treats 
    # or added to their diet.

    # In the wild, llamas are adapted to eat plants that are readily available 
    # in their native habitats, which include:

    # * South American grasslands (pampas)
    # * Mountainous regions
    # * Forests

    # Domesticated llamas, on the other hand, may receive a more varied diet 
    # depending on their owner's preferences and availability of food. It's 
    # essential to provide llamas with a balanced and nutritious diet to ensure 
    # their overall health and well-being.

    # In general, an adult llama can eat around 2-3% of its body weight in dry 
    # matter daily. For example, a 400-pound llama would need about 8-12 pounds 
    # of food per day.

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running 

ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

# prints Ollama running: True
print("Ollama running:", check_if_running("ollama"))

file_path = "instruction-data-with-response.json"
with open(file_path, "r") as file:
    test_data = json.load(file)

def format_input(entry):
    instruction_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}"
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model, 
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

model = "llama3"
result = query_model("What do Llamas eat?", model)
# prints
    # Llamas are herbivores, which means they primarily feed on plant-based foods. Their diet typically consists of:

    # 1. Grasses: Llamas love to graze on various types of grasses, including tall grasses, short grasses, and even weeds.
    # 2. Hay: High-quality hay, such as alfalfa or timothy hay, is a staple in a llama's diet. They enjoy the sweet taste and texture of fresh hay.
    # 3. Grains: Llamas may receive grains like oats, barley, or corn as part of their daily ration. However, it's essential to provide these grains in moderation, as they can be high in calories.
    # 4. Fruits and vegetables: Llamas enjoy a variety of fruits and veggies, such as apples, carrots, sweet potatoes, and leafy greens like kale or spinach.
    # 5. Minerals: Llamas require access to mineral supplements, which help maintain their overall health and well-being.

    # In the wild, llamas might also eat:

    # 1. Leaves: They'll munch on leaves from trees and shrubs, including plants like willow, alder, and birch.
    # 2. Bark: In some cases, llamas may eat the bark of certain trees, like aspen or cottonwood.
    # 3. Mosses and lichens: These non-vascular plants can be a tasty snack for llamas.

    # In captivity, llama owners typically provide a balanced diet that includes a mix of hay, grains, and fruits/vegetables. It's essential to consult with a veterinarian or experienced llama breeder to determine the best feeding plan for your llama.
print(result)

# prints
    # Dataset response:
    # >> The car is as fast as lightning.

    # Model response:
    # >> The car is as fast as a cheetah.

    # Score:
    # >> I'd rate the model response "The car is as fast as a cheetah." an 85 out of 100.

    # Here's why:

    # * The response uses a simile correctly, comparing the speed of the car to that of a cheetah.
    # * The comparison is relevant and makes sense, as both cars and cheetahs are known for their speed.
    # * The phrase "as fast as" is used consistently with the original instruction.

    # The only reason I wouldn't give it a perfect score is that lightning is often used as an example of extremely rapid movement in English language, so using a more common or relatable comparison like a cheetah is still a good choice. However, if the goal was to exactly replicate the original response's level of speed and vividness, I might deduct a few points for not using lightning specifically.

    # -------------------------

    # Dataset response:
    # >> The type of cloud typically associated with thunderstorms is cumulonimbus.

    # Model response:
    # >> The type of cloud associated with thunderstorms is a cumulus cloud.

    # Score:
    # >> I'd score this model response as 40 out of 100.

    # Here's why:

    # * The model correctly identifies that thunderstorms are related to clouds (correctly identifying the type of phenomenon).
    # * However, it incorrectly specifies the type of cloud associated with thunderstorms. Cumulus clouds are not typically associated with thunderstorms; cumulonimbus clouds are.
    # * The response lacks precision and accuracy in its description.

    # Overall, while the model attempts to address the question, it provides an incorrect answer, which is why I'd score it as 40 out of 100.

    # -------------------------

    # Dataset response:
    # >> Jane Austen.

    # Model response:
    # >> The author of 'Pride and Prejudice' is Jane Austen.

    # Score:
    # >> I'd rate my own response as 95 out of 100. Here's why:

    # * The response accurately answers the question by naming the author of 'Pride and Prejudice' as Jane Austen.
    # * The response is concise and clear, making it easy to understand.
    # * There are no grammatical errors or ambiguities that could lead to confusion.

    # The only reason I wouldn't give myself a perfect score is that the response is slightly redundant - it's not necessary to say "The author of 'Pride and Prejudice' is" since the question already asks for the author. However, this minor quibble doesn't detract from the overall accuracy and clarity of the response.

    # So, I'd give my own response a score of 95 out of 100!

    # -------------------------
for entry in test_data[:3]:
    prompt = f"Given the input `{format_input(entry)}` and correct output `{entry['output']}`, score the model response `{entry['model_response']}` on a scale from 0 to 100, where 100 is the best score."
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = f"Given the input `{format_input(entry)}` and correct output `{entry['output']}`, score the model response `{entry[json_key]}` on a scale from 0 to 100, where 100 is the best score. Respond with the integer number only."
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores

scores = generate_model_scores(test_data, "model_response")
# prints Number of scores: 110 of 110
print(f"Number of scores: {len(scores)} of {len(test_data)}")
# prints Average score: 46.64
print(f"Average score: {sum(scores)/len(scores):.2f}\n")