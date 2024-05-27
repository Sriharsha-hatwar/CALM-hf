import torch
from torch.utils.data import DataLoader
from transformers import (GPT2LMHeadModel,
                        GPT2Tokenizer,
                        AutoTokenizer,
                        AutoConfig
                        )
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from final_calm import CALMForCausalModeling


class CustomTrainer:
    def __init__(self, model, train_feats, val_feats, test_feats, batch_size=32, lr=1e-5, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders(train_feats, 
                                                    val_feats, 
                                                    test_feats, 
                                                    batch_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []

    def create_data_loaders(self, train_feats, val_feats, test_feats, batch_size):
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_feats["input_ids"]),
            torch.tensor(train_feats["attention_mask"])
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_feats["input_ids"]),
            torch.tensor(val_feats["attention_mask"])
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_feats["input_ids"]),
            torch.tensor(test_feats["attention_mask"])
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, val_loader, test_loader

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.eval_model(self.val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def train_epoch(self):
        print("Starting epoch")
        self.model.train()
        total_loss = 0.0
        data_iter = iter(self.train_loader)
        num_batches = len(self.train_loader)
        halfway_point = num_batches // 2
        num_tokens = 0
        total_perplexity = 0.0
        pbar = tqdm(total=num_batches, desc="Training")
        for batch_idx, (input_ids, attention_mask) in enumerate(data_iter):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = input_ids.clone().to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            perplexity = self.calculate_perplexity(logits, labels)
            total_perplexity += perplexity * labels.size(0)
            num_tokens += labels.size(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            pbar.update(1)
            # if batch_idx == halfway_point:
            #     val_loss = self.eval_model(self.val_loader)
            #     self.train_losses.append(total_loss * 2 /len(self.train_loader))
            #     self.val_losses.append(val_loss)
            #     print(f"Halfway through training, Validation Loss: {val_loss:.4f}")
        pbar.close()
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_tokens
        self.train_perplexities.append(avg_perplexity)
    
        return avg_loss

    def eval_model(self, dataset_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataset_loader)
        pbar = tqdm(total=num_batches, desc="Validation")
        num_tokens = 0
        total_perplexity = 0.0
        with torch.no_grad():
            for input_ids, attention_mask in dataset_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = input_ids.clone().to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                num_tokens += labels.size(0)
                perplexity = self.calculate_perplexity(logits, labels)
                total_perplexity += perplexity * labels.size(0)
                total_loss += loss.item()
                pbar.update(1)
        pbar.close()
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_tokens
        if dataset_loader == self.val_loader:
            self.val_perplexities.append(avg_perplexity)
        return avg_loss
    
    def calculate_perplexity(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = loss.exp().mean().item()
        return perplexity

    def test(self):
        test_loss = self.eval_model(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}")

# Example usage
def plot_loss(train_losses, valid_losses):
    # Assume your list of losses is called 'losses'

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the losses
    ax.plot(range(len(train_losses)), train_losses, marker='o', markersize=8, linestyle='-', linewidth=2, color='r', label='Train Losses')

    # Plot the second loss curve
    ax.plot(range(len(valid_losses)), valid_losses, marker='s', markersize=8, linestyle='--', linewidth=2, color='g', label='Valid Losses')
    # Set the title and axis labels


    ax.set_title('Train vs Valid loss', fontsize=18)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)

    # Add a legend
    ax.legend(fontsize=12)

    # Set the x-axis tick labels
    ax.set_xticks(range(len(train_losses)))
    ax.set_xticklabels([str(i) for i in range(len(train_losses))], fontsize=12, rotation=0)

    # num_ticks = int(max(range(len(train_losses))) / 0.5) + 1
    # xtick_positions = np.arange(0, len(train_losses), 0.5)
    # xtick_labels = [f'{i:.1f}' for i in xtick_positions]
    # ax.set_xticks(xtick_positions)
    # ax.set_xticklabels(xtick_labels, fontsize=12, rotation=45)

    # Set the y-axis tick labels
    ax.set_yticklabels([f'{loss:.2f}' for loss in ax.get_yticks()], fontsize=12)

    # Add grid lines
    ax.grid(linestyle='--', alpha=0.5)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')


def plot_perplexity(trainer):
    # Get the training and validation perplexities from the trainer
    train_perplexities = trainer.train_perplexities
    val_perplexities = trainer.val_perplexities

    fig, ax = plt.subplots(figsize=(8, 6))


    ax.plot(range(len(train_perplexities)), train_perplexities, marker='o', markersize=8, linestyle='-', linewidth=2, color='r', label='Training Perplexity')


    ax.plot(range(len(val_perplexities)), val_perplexities, marker='s', markersize=8, linestyle='--', linewidth=2, color='g', label='Validation Perplexity')

    ax.set_title('Perplexity Curve', fontsize=18)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Perplexity', fontsize=14)

    ax.legend(fontsize=12)

    ax.set_xticks(range(len(train_perplexities)))
    ax.set_xticklabels([str(i+1) for i in range(len(train_perplexities))], fontsize=12, rotation=45)

    ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('perplexity_curve.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    print("Running CALM model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment_model = "google/gemma-2b-it"
    anchor_model = "google/gemma-2b-it"
    num_cross_over = 2
    quantize = True
    calm_config = AutoConfig.from_pretrained("google/gemma-2b-it")
    model = CALMForCausalModeling(
        anchor_model,
        augment_model,
        num_cross_over,
        quantize,
        calm_config
    ).to(device)
    max_length = model.get_anchor_model().config.max_position_embeddings
    train_csv = pd.read_csv("gsm8k/main_train.csv")
    test_csv = pd.read_csv("gsm8k/main_test.csv")
    training_examples = ["Here is a Math question : "+ q + " Answer: " + a for q, a in zip(train_csv["question"], train_csv["answer"])]
    test_examples = ["Here is a Math question : "+ q + " Answer: " + a for q, a in zip(test_csv["question"], test_csv["answer"])]
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    train_encodings = tokenizer(training_examples, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_examples, truncation=True, padding=True, max_length=max_length)
    train_feats = {}
    val_feats = {}
    last_index = int(len(train_encodings["input_ids"]) * 0.9)
    train_feats["input_ids"] = train_encodings["input_ids"][:last_index]
    train_feats["attention_mask"] = train_encodings["attention_mask"][:last_index]
    val_feats["input_ids"] = train_encodings["input_ids"][last_index:]
    val_feats["attention_mask"] = train_encodings["attention_mask"][last_index:]
    
    #model = GPT2LMHeadModel.from_pretrained('gpt2').to(device) 
    
    trainer = CustomTrainer(model, train_feats, val_feats, test_encodings, batch_size=2, lr=1e-5, device=device)
    trainer.test()
    trainer.train(num_epochs=4)
    trainer.test()
    train_losses = trainer.train_losses
    valid_losses = trainer.val_losses
    if len(train_losses) != len(valid_losses):
        print("Straight to Jail.")
    plot_loss(train_losses, valid_losses)
    plot_perplexity(trainer)
    