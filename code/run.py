from dataclasses import dataclass, field


@dataclass
class Run:
    n: int
    m: int
    sequence_length: int
    d: int
    vocab_size: int = 0
    unique_beginnings: int = 0
    training_dataset: any = None
    model_obj: any = None
    model_num_params: int = 0
    training_loss_values: any = field(default_factory=list)
    emp_loss: float = 0.0
