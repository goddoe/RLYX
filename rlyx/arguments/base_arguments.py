import yaml
from dataclasses import dataclass, field


@dataclass
class BaseArgs:
    exp_name: str = "exp"
    logging_methods: list = field(default_factory=list)
    wandb_project: str = "default"
    wandb_entity: str = "default"

    # Dataset Args
    dataset_loader_name: str = "huggingface_loader"  # Dataset loader module
    dataset_name_or_path: str = "AI-MO/NuminaMath-TIR"  # HF dataset name or file path
    tokenized_dataset_path: str = "./data/NuminaMath-TIR_tokenized"
    overwrite_preprocess: bool = False
    train_size_limit: int = -1  # -1 means use all data
    valid_size_limit: int = -1  # -1 means use all data

    # Preprocessing Args
    batch_size_for_preproc: int = 3000
    num_proc_for_preproc: int = 16

    # Model Args
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M"
    
    # Experiment Module Args
    chat_template_name: str = None  # None to use tokenizer's default template
    tokenizer_function_name: str = "basic_tokenizer"
    evaluator_name: str = "basic_evaluator"
    reward_function_names: list = field(default_factory=lambda: ["format_reward"])

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
    stop_tokens: list = field(default_factory=lambda: ["<|im_end|>"])  # List of stop tokens for generation

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
