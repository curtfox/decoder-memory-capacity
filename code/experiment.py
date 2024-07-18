from dataclasses import dataclass, field
from run import Run
from haven import haven_utils as hu
import pickle
from nltk.tokenize import wordpunct_tokenize
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
import os


@dataclass
class Experiment:
    dataset: str
    batch_size: any
    epochs: int
    runs: list[Run] = field(default_factory=list)

    def run_experiment(self, plot_only, path, device):
        experiment_id = hu.hash_dict({"experiment": self})
        if plot_only == False:
            print("Run Experiment")
            full_dataset = self.process_data()
            emp_loss_dict = {}
            for run_num, run in enumerate(self.runs):
                torch.manual_seed(0)  # change torch seed here
                run.vocab_size, run.training_dataset = self.tokenize_data(
                    full_dataset, run
                )
                print("-----Run " + str(run_num + 1) + "-----")
                ### Train model
                run.model_obj, run.model_num_params = self.create_model(run, device)
                if not (str(run.n) in emp_loss_dict):
                    print("---Compute Empirical Loss---")
                    run.emp_loss, run.unique_beginnings = self.compute_empirical_loss(
                        run
                    )
                    emp_loss_dict[str(run.n)] = (run.emp_loss, run.unique_beginnings)
                else:
                    run.emp_loss = emp_loss_dict[str(run.n)][0]
                    run.unique_beginnings = emp_loss_dict[str(run.n)][1]
                print("Empirical Loss:", run.emp_loss)
                print("Unique Beginnings:", run.unique_beginnings)

                print("---Training---")
                run.training_loss_values = self.train(run, device)
            with open(
                os.path.join(path, "experiments", str(experiment_id) + ".pkl"),
                "wb",
            ) as f:
                pickle.dump({"experiment": self}, f)
            f.close()
        with open(
            os.path.join(path, "experiments", str(experiment_id) + ".pkl"), "rb"
        ) as f:
            experiment = pickle.load(f)
        f.close()
        ### Plot results
        print("Plot Experiment")
        print(experiment["experiment"])
        plot_experiment(experiment["experiment"], path)

    def process_data(self):
        np.random.seed(0)  # change numpy seed here
        if self.dataset == "tinystories":
            full_dataset = load_dataset("roneneldan/TinyStories", split="train")["text"]
            full_dataset = np.array(full_dataset)
            np.random.shuffle(full_dataset)
            full_dataset = full_dataset.tolist()
        return full_dataset

    def tokenize_data(self, full_dataset, run):
        ### Tokenize data
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

        datasetIDs = torch.tensor(datasetIDs)

        training_dataset = CustomTextDataset(sequence=datasetIDs)
        return vocab_size, training_dataset

    def create_model(self, run, device):
        model = DecoderOnlyTransformer(
            omega=run.vocab_size,
            d=run.d,
            m=run.m,
            tao=run.sequence_length,
            device=device,
        ).to(device)
        summary(model)
        model_num_params = sum(p.numel() for p in model.parameters())
        return model, model_num_params

    def compute_empirical_loss(self, run):

        for whole_dataset in torch_data.DataLoader(
            run.training_dataset, batch_size=run.n, shuffle=False
        ):
            training_dataset = whole_dataset

        num_data_points = training_dataset.size(0)
        vocab_size = run.vocab_size
        print("Vocab Size:", vocab_size)
        seq_length = run.sequence_length

        emp_loss = 0
        unique_beginnings = 0
        # for each length (1 to seq_length - 1)
        for t in range(1, seq_length):
            unique_rows, inverse, counts = torch.unique(
                training_dataset[:, 0:t],
                sorted=False,
                return_inverse=True,
                return_counts=True,
                dim=0,
            )
            num_unique_rows = unique_rows.size(0)
            unique_beginnings += num_unique_rows

            # for each unique context
            for i in range(num_unique_rows):
                # for each token in vocab
                for gamma in range(vocab_size):
                    pi_hat = 0
                    # for each data point
                    for j in range(num_data_points):
                        # check if data sequence (beginning) j corresponds to unique context i
                        # AND
                        # check if next token in data sequence (beginning) j equals token gamma
                        if inverse[j] == i and training_dataset[j, t] == gamma:
                            pi_hat += 1

                    pi_hat = pi_hat / counts[i]
                    # check if pi_hat == 0
                    if pi_hat != 0:
                        emp_loss += -counts[i] * pi_hat * torch.log(pi_hat)

        return emp_loss.item(), unique_beginnings

    def train(self, run, device):
        if self.batch_size == "full":
            batch_size = run.n
        else:
            batch_size = self.batch_size

        criterion = nn.CrossEntropyLoss(reduction="sum")
        step_size = 0.0001
        optimizer = optim.Adam(run.model_obj.parameters(), lr=step_size)
        ### Run Training loop
        trainloader = torch_data.DataLoader(
            run.training_dataset, batch_size=batch_size, shuffle=False
        )
        training_loss_vals = []
        for epoch in range(self.epochs):
            for _, sequence_batch in enumerate(trainloader):
                sequence_batch = sequence_batch.to(device)
                optimizer.zero_grad()
                output = run.model_obj(sequence_batch[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, run.vocab_size),
                    sequence_batch[:, 1:].contiguous().view(-1),
                )
                loss.backward()
                optimizer.step()
            full_loss = compute_full_training_loss(run, device, batch_size)
            if epoch == 0:
                print(f"Epoch: {epoch}, Loss: {full_loss}")
                training_loss_vals.append(full_loss)
            if (epoch + 1) % 5000 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {full_loss}")
                training_loss_vals.append(full_loss)

        print(f"Final Epoch Loss: {full_loss}")
        print(f"Entropy Lower Bound: {run.emp_loss}")
        print(f"Difference: {full_loss - run.emp_loss}")
        return training_loss_vals


def compute_full_training_loss(run, device, batch_size):
    criterion = nn.CrossEntropyLoss(reduction="sum")

    full_loss = 0
    for sequence_batch in torch_data.DataLoader(
        dataset=run.training_dataset,
        batch_size=batch_size,
        shuffle=False,
    ):
        sequence_batch = sequence_batch.to(device)
        output = run.model_obj(sequence_batch[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, run.vocab_size),
            sequence_batch[:, 1:].contiguous().view(-1),
        )
        full_loss += loss

    full_loss = full_loss
    return full_loss.item()
