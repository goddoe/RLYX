import yaml
from dataclasses import dataclass, field


@dataclass
class BaseArgs:
    exp_name: str = "exp"
    logging_methods: list = field(default_factory=list)
    wandb_project: str = "default"
    wandb_entity: str = "default"

    # Dataset Args
    dataset_name_or_path: str = "AI-MO/NuminaMath-TIR"
    tokenized_dataset_path: str = "./data/NuminaMath-TIR_tokenized"
    overwrite_preprocess: bool = False

    # Preprocessing Args
    batch_size_for_preproc: int = 3000
    num_proc_for_preproc: int = 16

    # Model Args
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M"

    # Training Args
    max_length: int = 1024
    num_train_epochs: int = 3
    num_warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 1.e-5
    max_grad_value: float = 1.0
    train_batch_size_per_proc: int = 2
    eval_batch_size_per_proc: int = 2
    gradient_accumulation_steps: int = 1

    # Rollout Args
    rollout_per_sample: int = 3
    rollout_temperature: float = 1.0
    rollout_max_tokens: int = 512
    kl_coef: float = 0.01

    eval_interval: int = 500
    log_interval: int  = 100
    save_interval: int = 500
    save_dir: str ='./ckpts/exp'

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return str(self.to_dict())
