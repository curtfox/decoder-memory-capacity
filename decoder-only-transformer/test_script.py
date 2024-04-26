import torch

# data = torch.randint(1, 10, (10, 5))  # (batch_size, seq_length)
data = torch.tensor([[1, 3, 4, 0], [1, 3, 2, 4]])
num_data_points = data.size(0)
seq_length = data.size(1)
vocab = torch.unique(data, sorted=False, return_inverse=False, return_counts=False)
vocab_size = len(vocab)

emp_loss = 0
# for each length (1 to seq_length - 1)
for t in range(1, seq_length):
    unique_rows, inverse, counts = torch.unique(
        data[:, 0:t], sorted=False, return_inverse=True, return_counts=True, dim=0
    )
    num_unique_rows = unique_rows.size(0)

    print("Length:", t)
    print("Original Rows:\n", data[:, :t])
    print("Unique Rows:\n", unique_rows)
    print("Counts:\n", counts)
    print("Inverse:\n", inverse)

    # print("Begin Loop")
    # for each unique row
    for i in range(num_unique_rows):
        # print("Unique row:", i)
        # for each token in vocab
        for gamma in range(vocab_size):
            # print("Token:", gamma)
            pi_hat = 0
            # for each data point
            for j in range(num_data_points):
                # print("Data Point:", j)
                # check if data sequence (beginning) k corresponds to unique sequence beginning i
                # AND
                # check if next token in data sequence (beginning) k equals token gamma
                if inverse[j] == i and data[j, t] == gamma:
                    pi_hat = pi_hat + 1

            pi_hat = pi_hat / counts[i]
            # check if pi_hat == 0
            if pi_hat != 0:
                emp_loss = emp_loss + -counts[i] * pi_hat * torch.log(pi_hat)

print("Empirical Loss:", emp_loss.item())

print(
    (
        -torch.log(torch.tensor(1))
        - 2 * (1 / 2) * torch.log(torch.tensor(1 / 2))
        - 2 * (1 / 2) * torch.log(torch.tensor(1 / 2))
        - torch.log(torch.tensor(1))
        - torch.log(torch.tensor(1))
    ).item()
)
