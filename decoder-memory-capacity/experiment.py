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
    batch_size: any
    epochs: int
    runs: list[Run] = field(default_factory=list)

    def run_experiment(self, plot_only, path, device):
        experiment_id = hu.hash_dict({"experiment": self})
        if plot_only == False:
            print("Run Experiment")
            np.random.seed(0)
            full_dataset = load_dataset("roneneldan/TinyStories", split="train")["text"]
            full_dataset = np.array(full_dataset)
            # print("Before shuffle\n")
            # print(full_dataset[0:2])
            np.random.shuffle(full_dataset)
            # print("After shuffle\n")
            # print(full_dataset[0:2])
            full_dataset = full_dataset.tolist()
            # print("Back to list\n")
            # print(full_dataset[0:2])
            ### Process data
            """
            emp_loss_dict = {
                "100": 351.5614624,  # 283
                "200": 754.4559326,  # 649
                "300": 1172.5532227,  # 979
                "400": 1629.6152344,  # 1258
                "500": 2131.4968262,  # 1559
                "600": 2685.7810059,  # 1838
                "700": 3189.8449797,  # 1929
                "800": 3459.7507324,  # 2508
                "900": 3818.5856934,  # 2517
                "1000": 4180.9946289,  # 2932
            }
            """
            emp_loss_dict = {}
            for run_num, run in enumerate(self.runs):
                torch.manual_seed(0)
                run.vocab_size, run.training_dataset = self.process_data(
                    full_dataset, run
                )
                print("-----Run " + str(run_num + 1) + "-----")
                ### Train model
                run.model_obj, run.model_num_params = self.create_model(run, device)
                if not (str(run.n) in emp_loss_dict):
                    print("---Compute Empirical Loss---")
                    emp_loss_dict[str(run.n)], run.unique_beginnings = (
                        self.compute_empirical_loss(run)
                    )
                    run.emp_loss = emp_loss_dict[str(run.n)]
                else:
                    run.emp_loss = emp_loss_dict[str(run.n)]
                print("Empirical Loss:", emp_loss_dict[str(run.n)])

                print("---Training---")
                run.training_loss_values = self.train(run, device)
            with open(
                os.path.join(path, "experiments", str(experiment_id) + ".pkl"),
                "wb",
            ) as f:
                pickle.dump({"experiment": self}, f)
            f.close()
            print("Empirical Loss Dict")
            print(emp_loss_dict)
        with open(
            os.path.join(path, "experiments", str(experiment_id) + ".pkl"), "rb"
        ) as f:
            experiment = pickle.load(f)
        f.close()
        print(experiment["experiment"])
        ### Plot results
        print("Plot Experiment")
        plot_experiment(experiment["experiment"], path)

    def process_data(self, full_dataset, run):
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

        # vocab_size = 200
        # datasetIDs = torch.randint(1, vocab_size, (run.n, run.sequence_length))

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

            # for each unique row
            for i in range(num_unique_rows):
                # for each token in vocab
                for gamma in range(vocab_size):
                    pi_hat = 0
                    # for each data point
                    for j in range(num_data_points):
                        # check if data sequence (beginning) j corresponds to unique sequence beginning i
                        # AND
                        # check if next token in data sequence (beginning) j equals token gamma
                        if inverse[j] == i and training_dataset[j, t] == gamma:
                            pi_hat += 1

                    pi_hat = pi_hat / counts[i]
                    # check if pi_hat == 0
                    if pi_hat != 0:
                        emp_loss += -counts[i] * pi_hat * torch.log(pi_hat)

        print("Unique Beginnings:", unique_beginnings)
        # print(emp_loss.item())
        return emp_loss.item(), unique_beginnings

    def train(self, run, device):
        if self.batch_size == "full":
            batch_size = run.n
        else:
            batch_size = self.batch_size

        criterion = nn.CrossEntropyLoss(reduction="sum")

        optimizer = optim.Adam(run.model_obj.parameters(), lr=0.01)

        ### Run Training loop
        trainloader = torch_data.DataLoader(
            run.training_dataset, batch_size=batch_size, shuffle=False
        )
        training_loss_vals = []
        step_size_decreased_1 = False
        step_size_decreased_2 = False
        step_size_decreased_3 = False
        for epoch in range(self.epochs):
            for _, sequence_batch in enumerate(trainloader):
                sequence_batch = sequence_batch.to(device)
                optimizer.zero_grad()
                output = run.model_obj(sequence_batch[:, :-1])  # sequence_batch[:, :-1]
                loss = criterion(
                    output.contiguous().view(-1, run.vocab_size),
                    sequence_batch[:, 1:]
                    .contiguous()
                    .view(-1),  # sequence_batch[:, 1:]
                )
                loss.backward()
                optimizer.step()
            full_loss = compute_full_training_loss(run, device, batch_size)
            if (full_loss - run.emp_loss < 100) and (not step_size_decreased_1):
                print("Decrease step size first")
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                step_size_decreased_1 = True
            if (full_loss - run.emp_loss < 10) and (not step_size_decreased_2):
                print("Decrease step size second")
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                step_size_decreased_2 = True
            if (full_loss - run.emp_loss < 1) and (not step_size_decreased_3):
                print("Decrease step size third")
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                step_size_decreased_3 = True
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch: {epoch+1}, Loss: {full_loss}")
                training_loss_vals.append(full_loss)

        print(f"Final Epoch Loss: {full_loss}")
        print(f"Empirical Loss: {run.emp_loss}")
        print(f"Absolute Difference: {full_loss - run.emp_loss}")
        print(f"Relative Difference: {(full_loss - run.emp_loss)/run.emp_loss}")
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
