from dataclasses import dataclass, field
from run import Run
from haven import haven_utils as hu
import pickle
from nltk.tokenize import wordpunct_tokenize, TreebankWordTokenizer
from gensim.corpora.dictionary import Dictionary
import torch
from customTextDataset import *
from transformer_model import *
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from plotting import *
from datasets import load_dataset


@dataclass
class Experiment:
    path: str
    batch_size: any
    epochs: int
    device: str = "cuda"
    ### Lists
    runs: list[Run] = field(default_factory=list)

    def run_experiment(self, plot_only):
        experiment_id = hu.hash_dict({"experiment": self})
        # print("Experiment ID: ", experiment_id)
        if plot_only == False:
            print("Run Experiment")
            full_dataset = load_dataset("roneneldan/TinyStories", split="train")["text"]
            ### Process data
            for run_num, run in enumerate(self.runs):
                torch.manual_seed(0)
                run.vocab_size, run.training_dataset = self.process_data(
                    full_dataset, run
                )
                print("-----Run " + str(run_num + 1) + "-----")
                ### Train model
                run.model_obj, run.model_num_params = self.create_model(run)
                print("---Training---")
                run.training_loss_values = self.train(run)

            with open(
                self.path + "/experiments/" + str(experiment_id) + ".pkl", "wb"
            ) as f:
                pickle.dump({"experiment": self}, f)
            f.close()

        with open(self.path + "/experiments/" + str(experiment_id) + ".pkl", "rb") as f:
            experiment = pickle.load(f)
        f.close()
        # print(experiment)
        ### Plot results
        print("Plot Experiment")
        plot_experiment(experiment["experiment"])

    def process_data(self, full_dataset, run):
        ### Read in data
        # dataset.save_to_disk(path + "/data/" + "TinyStoriesTrain.txt")
        # dataset = datasets.load_from_disk(path + "/data/" + "TinyStoriesTrain.txt")

        ### Tokenize data
        # tokenizer = TreebankWordTokenizer()
        # tokenizer.tokenize()
        datasetTokens = []
        j = 0
        for _, story in enumerate(full_dataset):
            tokenized_story = wordpunct_tokenize(story)
            if len(tokenized_story) >= run.sequence_length:
                tokenized_story = tokenized_story[: run.sequence_length]
                datasetTokens.append(tokenized_story)
                j = j + 1
                if j == run.n:
                    break

        vocab = Dictionary(datasetTokens)
        vocab_size = len(Dictionary(datasetTokens))

        ### Convert tokens to ID's
        datasetIDs = []
        for story in datasetTokens:
            storyID = []
            for word in story:
                storyID.append(vocab.token2id[word])
            datasetIDs.append(storyID)

        training_dataset = CustomTextDataset(sequence=torch.tensor(datasetIDs))
        return vocab_size, training_dataset

    def create_model(self, run):
        model = DecoderOnlyTransformer(
            omega=run.vocab_size,
            d=run.d,
            num_heads=1,
            num_decode_layers=1,
            m=run.m,
            tao=run.sequence_length,
            device=self.device,
        ).to(self.device)
        summary(model)
        model_num_params = sum(p.numel() for p in model.parameters())
        return model, model_num_params

    def train(self, run):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(
            run.model_obj.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )

        ### Run Training loop
        trainloader = torch_data.DataLoader(
            run.training_dataset, batch_size=self.batch_size, shuffle=True
        )
        training_loss_vals = []
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch+1}")
            for _, sequence_batch in enumerate(trainloader):
                # print(f"Batch: {i+1}")
                sequence_batch = sequence_batch.to(self.device)
                optimizer.zero_grad()
                # print("Input Data Size:", sequence_batch.size())
                output = run.model_obj(sequence_batch[:, :-1])  # sequence_batch[:, :-1]
                # print("Output Size:", output.contiguous().view(-1, vocab_size).size())
                loss = criterion(
                    output.contiguous().view(-1, run.vocab_size),
                    sequence_batch[:, 1:]
                    .contiguous()
                    .view(-1),  # sequence_batch[:, 1:]
                )
                loss.backward()
                optimizer.step()
            full_loss = self.compute_full_training_loss(run)
            training_loss_vals.append(full_loss)
            print(f"Epoch: {epoch+1}, Loss: {full_loss}")

        return training_loss_vals

    def compute_full_training_loss(self, run):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        full_loss = 0
        for sequence_batch in torch_data.DataLoader(
            dataset=run.training_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        ):
            sequence_batch = sequence_batch.to(self.device)
            output = run.model_obj(sequence_batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, run.vocab_size),
                sequence_batch[:, 1:].contiguous().view(-1),
            )
            full_loss = full_loss + loss * len(sequence_batch)

        full_loss = full_loss / run.n
        return full_loss.item()
