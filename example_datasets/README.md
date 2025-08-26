# Example Datasets

이 디렉토리는 custom dataset loader를 테스트하기 위한 예제 데이터셋을 포함합니다.

## 디렉토리 구조
```
example_datasets/
├── json/         # JSON 형식 예제
│   └── example.json
├── csv/          # CSV 형식 예제
│   ├── train.csv
│   └── test.csv
├── jsonl/        # JSONL 형식 예제
│   ├── train.jsonl
│   └── test.jsonl
└── text/         # 텍스트 파일 형식 예제
    ├── train.txt
    └── test.txt
```

## 사용 예시

### JSON 데이터셋 사용
```yaml
dataset_loader_name: "json_loader"
dataset_name_or_path: "./example_datasets/json/example.json"
```

### CSV 데이터셋 사용
```yaml
dataset_loader_name: "csv_loader"
dataset_name_or_path: "./example_datasets/csv"
```

### JSONL 데이터셋 사용
```yaml
dataset_loader_name: "jsonl_loader"
dataset_name_or_path: "./example_datasets/jsonl"
```

### 텍스트 파일 데이터셋 사용
```yaml
dataset_loader_name: "text_file_loader"
dataset_name_or_path: "./example_datasets/text"
```

## 테스트 config 예시

```yaml
exp_name: "test_custom_dataset"
logging_methods: ["tensorboard"]

# Dataset Args
dataset_loader_name: "json_loader"
dataset_name_or_path: "./example_datasets/json/example.json"
tokenized_dataset_path: "./data/custom_tokenized"
overwrite_preprocess: true
train_size_limit: -1
valid_size_limit: -1

# Model Args
model_name_or_path: "HuggingFaceTB/SmolLM2-135M"

# Experiment Module Args
chat_template_name: null  # Use model's default template
tokenizer_function_name: "basic_tokenizer"
evaluator_name: "simple_evaluator"

# Training Args
max_length: 512
num_train_epochs: 1
train_batch_size_per_proc: 1
eval_batch_size_per_proc: 1
```