import urllib.request
import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# downloading the verdict 

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99]) # prints the first 100 chars

# using regular expressions

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
# prints ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
print(result)

result = re.split(r'([,.]|\s)', text)
# prints ['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']
print(result)

result = [item for item in result if item.strip()]
# prints ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
# prints ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
print(result)

preproccessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preproccessed = [item.strip() for item in preproccessed if item.strip()]
# prints 4690
print(len(preproccessed))
# prints ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
print(preproccessed[:30])

all_words = sorted(set(preproccessed))
vocab_size = len(all_words)
# prints 1130
print(vocab_size)

# prints:
    # ('!', 0)
    # ('"', 1)
    # ("'", 2)
    # ('(', 3)
    # (')', 4)
    # (',', 5)
    # ('--', 6)
    # ('.', 7)
    # (':', 8)
    # (';', 9)
    # ('?', 10)
    # ('A', 11)
    # ('Ah', 12)
    # ('Among', 13)
    # ('And', 14)
    # ('Are', 15)
    # ('Arrt', 16)
    # ('As', 17)
    # ('At', 18)
    # ('Be', 19)
    # ('Begin', 20)
    # ('Burlington', 21)
    # ('But', 22)
    # ('By', 23)
    # ('Carlo', 24)
    # ('Chicago', 25)
    # ('Claude', 26)
    # ('Come', 27)
    # ('Croft', 28)
    # ('Destroyed', 29)
    # ('Devonshire', 30)
    # ('Don', 31)
    # ('Dubarry', 32)
    # ('Emperors', 33)
    # ('Florence', 34)
    # ('For', 35)
    # ('Gallery', 36)
    # ('Gideon', 37)
    # ('Gisburn', 38)
    # ('Gisburns', 39)
    # ('Grafton', 40)
    # ('Greek', 41)
    # ('Grindle', 42)
    # ('Grindles', 43)
    # ('HAD', 44)
    # ('Had', 45)
    # ('Hang', 46)
    # ('Has', 47)
    # ('He', 48)
    # ('Her', 49)
    # ('Hermia', 50)
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # inverse vocabulary that maps token ID back to the original text tokens

    def encode(self, text):
        preproccessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        ids = [self.str_to_int[s] for s in preproccessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # removes spaces before the specified punctuation
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# prints [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
print(ids)
# prints " It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
print(tokenizer.decode(ids))

text = "Hello, do you like tea?"
# would result in KeyError: 'Hello'
# print(tokenizer.encode(text))

all_tokens = sorted(list(set(preproccessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# prints 1132
print(len(vocab.items()))

# prints
    # ('younger', 1127)
    # ('your', 1128)
    # ('yourself', 1129)
    # ('<|endoftext|>', 1130)
    # ('<|unk|>', 1131)
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preproccessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        preproccessed = [item if item in self.str_to_int else "<|unk|>" for item in preproccessed] # replace unknown words with '<|unk|>' tokens
        ids = [self.str_to_int[s] for s in preproccessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # replace spaces before the specified punctuations
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# prints Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
print(text)

tokenizer = SimpleTokenizerV2(vocab)
# prints [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
print(tokenizer.encode(text))
# prints <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
print(tokenizer.decode(tokenizer.encode(text)))

# byte pair encoding

# prints tiktoken version: 0.8.0
print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# prints [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
print(integers)

strings = tokenizer.decode(integers)
# prints Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
print(strings)

exercise2_1_encoded = tokenizer.encode("Akwirw ier")
# prints [33901, 86, 343, 86, 220, 959]
print(exercise2_1_encoded)
exercise2_1_decoded = tokenizer.decode(exercise2_1_encoded)
# prints Akwirw ier
print(exercise2_1_decoded)

# data sampling with a sliding window

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text) 
# prints 5145
print(len(enc_text))

enc_sample = enc_text[50:] # remove the first 50 tokens
context_size = 4 # determines how many tokens are included in the input
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
# prints x: [290, 4920, 2241, 287]
print(f"x: {x}")
# prints y:      [4920, 2241, 287, 257]
print(f"y:      {y}")

# prints
    # [290] ----> 4920
    # [290, 4920] ----> 2241
    # [290, 4920, 2241] ----> 287
    # [290, 4920, 2241, 287] ----> 257
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# prints
    # and ---->  established
    # and established ---->  himself
    # and established himself ---->  in
    # and established himself in ---->  a
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) # tokenizes the entire text
        
        # use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids) 

    # returns a single row in the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, # drops the last batch if it's shorter than the specified batch_size to prevent loss spikes during training
        num_workers=num_workers
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) # converts dataloader into a Python iterator to fetch the next entry via the built-in next() function
first_batch = next(data_iter)
# prints [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
print(first_batch)

second_batch = next(data_iter)
# prints [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# prints
    # Inputs:
    # tensor([[   40,   367,  2885,  1464],
    #         [ 1807,  3619,   402,   271],
    #         [10899,  2138,   257,  7026],
    #         [15632,   438,  2016,   257],
    #         [  922,  5891,  1576,   438],
    #         [  568,   340,   373,   645],
    #         [ 1049,  5975,   284,   502],
    #         [  284,  3285,   326,    11]])
print("Inputs:\n", inputs)
# prints
    # Targets:
    # tensor([[  367,  2885,  1464,  1807],
    #         [ 3619,   402,   271, 10899],
    #         [ 2138,   257,  7026, 15632],
    #         [  438,  2016,   257,   922],
    #         [ 5891,  1576,   438,   568],
    #         [  340,   373,   645,  1049],
    #         [ 5975,   284,   502,   284],
    #         [ 3285,   326,    11,   287]])
print("\nTargets:\n", targets)

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# prints
    # Parameter containing:
    # tensor([[ 0.3374, -0.1778, -0.1690],
    #         [ 0.9178,  1.5810,  1.3010],
    #         [ 1.2753, -0.2010, -0.1606],
    #         [-0.4015,  0.9666, -1.1481],
    #         [-1.1589,  0.3255, -0.6315],
    #         [-2.8400, -0.7849, -1.4096]], requires_grad=True)
print(embedding_layer.weight)
# prints tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
print(embedding_layer(torch.tensor([3])))
# prints
    # tensor([[ 1.2753, -0.2010, -0.1606],
    #         [-0.4015,  0.9666, -1.1481],
    #         [-2.8400, -0.7849, -1.4096],
    #         [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
print(embedding_layer(input_ids))

# encoding word positions

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# prints
    # TokenIDs:
    # tensor([[   40,   367,  2885,  1464],
    #         [ 1807,  3619,   402,   271],
    #         [10899,  2138,   257,  7026],
    #         [15632,   438,  2016,   257],
    #         [  922,  5891,  1576,   438],
    #         [  568,   340,   373,   645],
    #         [ 1049,  5975,   284,   502],
    #         [  284,  3285,   326,    11]])
print("TokenIDs:\n", inputs)
# prints
    # Inputs shape:
    # torch.Size([8, 4])
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
# prints torch.Size([8, 4, 256])
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# prints torch.Size([4, 256])
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
# prints torch.Size([8, 4, 256])
print(input_embeddings.shape)