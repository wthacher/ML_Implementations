'''

Implement a modern GPT style model parallelized using Ray which can run on my macbook pro.

1.Download a Wikipedia dataset from Huggingface
2.Tokenize the dataset

'''
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


## 1: Data download and tokenization
# 1a. Configuration
# DATASET_NAME = "wikimedia/wikipedia"
# DATASET_CONFIG = "20231101.en"
DATASET_NAME = "roneneldan/TinyStories"
MODEL_NAME = "gpt2" # Using GPT-2's BPE vocabulary
SAVE_PATH = "./TinyStories_tokenized"
D_HIDDEN = 256
N_HEADS = 8
N_LAYERS = 4
MAX_LENGTH = 128
NUM_PROC = 8
# 1b. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
VOCAB_SIZE = 10000
print(f"Vocab size: {VOCAB_SIZE}")
# GPT2 doesn't have a pad token by default, so we add one
tokenizer.pad_token = tokenizer.eos_token

# 1c. Load the dataset, if needed
##first check if tokenized dataset is already on disk
if os.path.exists(SAVE_PATH):
    tokenized_dataset = load_from_disk(SAVE_PATH)
else:
    print("Loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME)
    #raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train[:10%]')

    # 1d. Tokenization function
    def tokenize_function(examples):
        # We truncate and pad to a fixed length (e.g., 512 or 1024)
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length"
        )

    # 1e. Process the whole thing in parallel
    print("Tokenizing...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=NUM_PROC
        #remove_columns=["text", "url", "title", "id"] # Drop raw text to save space
    )

    # 1f. Save to disk
    tokenized_dataset.save_to_disk(SAVE_PATH)
    print(f"Success! Tokenized dataset saved to {SAVE_PATH}")


import torch.nn as nn
import torch

##2. Build model


# 2a. build transformer block 
class TransformerBlock(nn.Module):
    def __init__(self,d_h: int, n_heads: int, d_out: int):
        super(TransformerBlock, self).__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_out = d_out
        ##initialize self-attention layer
        self.self_attention_layer = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        ##initialize feedforward layer
        self.feedforward_layer = nn.Sequential(
            nn.Linear(d_h, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_h)
        )
        ##initialize layer normalization layers
        self.layer_norm1 = nn.LayerNorm(d_h)
        self.layer_norm2 = nn.LayerNorm(d_h)

        ##initialize dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ##do attention layer computation
        x_norm = self.layer_norm1(x_in)
        x_attn, _ = self.self_attention_layer(x_norm, x_norm, x_norm, attn_mask=mask, 
            need_weights=False)
        
        x_attn = self.dropout(x_attn)
        x = x_in + x_attn
        ##do feedforward layer computation
        x_norm = self.layer_norm2(x)
        x_ff = self.feedforward_layer(x_norm)
        x_ff = self.dropout(x_ff)
        x = x + x_ff
        return x

# 2b. Add down projection, positional encoding, TBs, and up projection
class GPT_model(nn.Module):
    def __init__(self, d_vocab, d_hidden, n_heads, n_layers):
        assert d_hidden % n_heads == 0, "error: d_hidden must be divisible by n_heads"
        super(GPT_model, self).__init__()
        self.d_vocab = d_vocab
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.seq_len = MAX_LENGTH

        ##initialize embedding layer
        self.embedding_layer = nn.Embedding(d_vocab, d_hidden)
        self.pos_embedding = nn.Embedding(self.seq_len, d_hidden)

        ##initialize layer normalization layer
        ##initialize transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_hidden, n_heads, d_hidden) for _ in range(n_layers)])
        self.projection_layer = nn.Linear(d_hidden, d_vocab)

        #implement weight tying by just transposing the embedding layer weights

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        idx = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding_layer(x) + self.pos_embedding(idx) ##broadcast over batch

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for block in self.transformer_blocks:
            x = block(x, mask)
        
        logits = self.projection_layer(x)
        return logits


## 3: Training loop 
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from torch.utils.data import DataLoader
import tempfile


import ray
import torch
import time
# 3a. Make training function which each worker will use
def train_func():
    config={}
    config["lr"]=1e-4
    config["batch_size"]=64
    config["num_epochs"]=10


    use_ray = False
    device = torch.device("cpu")
    if not use_ray and torch.backends.mps.is_available():
        device = torch.device("mps")

    ##initialize model
    model = GPT_model(d_vocab= VOCAB_SIZE, d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS)


    if os.path.exists("gpt_model.pt"):
        model.load_state_dict(torch.load("gpt_model.pt", map_location=device))
    
    if use_ray:
        model = ray.train.torch.prepare_model(model)
    else:
        model = model.to(device)
    
    ##initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    ##initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    ##initialize data loader
    
    train_loader=None
    if use_ray:
        train_loader = DataLoader(tokenized_dataset["train"], batch_size=config["batch_size"], shuffle=True)
        train_loader = ray.train.torch.prepare_data_loader(train_loader)
        print(f"batches per epoch: {len(train_loader)}")
    

    ##load the entire tokenized dataset into memory on the device rather than using a data loader
    
    print("Loading data into memory.")
    start_time = time.time()
    subset_size = 50000 # Use a small number for testing
    test_subset = tokenized_dataset["train"].select(range(subset_size))

    # Convert to int32 (fastest for MPS) and move to device ONCE
    all_input_ids = torch.tensor(test_subset['input_ids'], dtype=torch.int32).to(device)
    all_masks = torch.tensor(test_subset['attention_mask'], dtype=torch.bool).to(device)

    # start_time = time.time()
    # all_input_ids = torch.tensor(tokenized_dataset['train']['input_ids'][0], dtype=torch.int32).to(device)
    # all_masks = torch.tensor(tokenized_dataset['train']['attention_mask'][0], dtype=torch.int32).to(device)
    end_time = time.time()
    print(f"Time taken to load data: {end_time - start_time} seconds")
    print(f"Data is ready on {device}. Size: {all_input_ids.element_size() * all_input_ids.nelement() / 1e9:.2f} GB")

    batch_size = config["batch_size"]
    
    
    num_samples = all_input_ids.size(0)
    for epoch in range(config["num_epochs"]):
        # if use_ray and ray.train.get_context().get_world_size() > 1:
        #     train_loader.sampler.set_epoch(epoch)
        indices = torch.randperm(num_samples, device=device)
        
        seq_in=None
        
        ##add timers to see what is slowing down the training  
        start_time = time.time()
        time_loop = False

        for i in range(0, num_samples, batch_size):
            if time_loop:
                torch.mps.synchronize()
                t0 = time.time()
            # Grab a batch of indices
            batch_indices = indices[i : i + batch_size]
            
            # Slice the data
            # input_ids needs to be .long() for the embedding layer inside the forward pass
            b_ids = all_input_ids[batch_indices].long() 
            b_mask = all_masks[batch_indices]
            
            seq_in = b_ids[:, :-1]
            targets = b_ids[:, 1:]
            
            if time_loop:
                torch.mps.synchronize()
                print(f"Time taken to load batch: {time.time() - t0} seconds")
                t0 = time.time()

            logits = model(seq_in) ##model outputs next token logits
            if time_loop:
                torch.mps.synchronize()
                print(f"Forward time: {time.time() - t0} seconds")
                t0 = time.time()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)).contiguous(), targets.reshape(-1).contiguous()) ##flatten for speed here
            
            if time_loop:
                torch.mps.synchronize()
                print(f"Loss time: {time.time() - t0} seconds")
                t0 = time.time()
            loss.backward()
            
            if time_loop:
                torch.mps.synchronize()
                print(f"Backward time: {time.time() - t0} seconds")
                t0 = time.time()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        
        
        end_time = time.time()       
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time taken for epoch {epoch}: {end_time - start_time} seconds")
        
        if use_ray:
            metrics = {"loss": loss.item(), "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pt")
                )
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )
            if ray.train.get_context().get_world_rank() == 0:
                print(metrics)
    
    if not use_ray:
        ##save model to disk
        torch.save(model.state_dict(), "gpt_model.pt")
        print("Model saved to disk")
    else:
        print("Model saved to ray checkpoint")
    return 

# train_func()
# import matplotlib.pyplot as plt
# plt.plot(loss_history)
# plt.show()


# scaling_config = ray.train.ScalingConfig(num_workers=8, use_gpu=False)
# trainer = ray.train.torch.TorchTrainer(
#     train_func,
#     scaling_config=scaling_config,
# )
# result = trainer.fit()

##load model from disk and write a sampling function to generate text       
device = torch.device("mps")
model = GPT_model(d_vocab= VOCAB_SIZE, d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS)
model.load_state_dict(torch.load("gpt_model.pt"))
model.to(device)

def sample_text(model, tokenizer, max_length=100):
    model.eval()
    tokens = tokenizer.encode("Tell me a story about a dog.")
    
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_length):
        logits = model(tokens)
        ##sample from multinomial distribution
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        #next_token = torch.argmax(logits[0, -1, :]) ##argmax sampler
        tokens = torch.cat([tokens, next_token.reshape(1,1)], dim=1)
    return tokenizer.decode(tokens[0].tolist())

print(sample_text(model, tokenizer))


##Next you need to write some evaluation functions