# Reward Functions Module

This module contains various reward functions that can be used in the training process. The reward system is now modular and can be configured through the experiment configuration files.

## Available Reward Functions

### 1. `format_reward`
- Validates the format of generated text
- Returns 0-1 based on format correctness
- Supports two format types: `format_type_1` and `format_type_2`

### 2. `transcribe_reward`
- Evaluates transcription quality using WER and CER metrics
- Uses exponential decay based on error rate
- Returns values between 0-1 with cutoff at 0.2

### 3. `number_reward`
- Compares numerical values in predicted and gold text
- Handles Korean numbers, Arabic numerals, and mixed formats
- Returns 1.0 for exact match (with optional tolerance), 0.0 otherwise

### 4. `combined_reward`
- Combines multiple rewards (currently format + transcribe)
- Returns the sum of individual rewards

## How to Use

1. **In Experiment Configuration (YAML)**:
```yaml
# Experiment Module Args
reward_function_names: ["format_reward", "transcribe_reward"]
```

2. **Available Options**:
- Single reward: `["format_reward"]`
- Multiple rewards: `["format_reward", "transcribe_reward"]`
- Combined reward: `["combined_reward"]`
- Number reward: `["number_reward"]`

## Creating New Reward Functions

To create a new reward function:

1. Create a new Python file in `rlyx/rewards/`
2. Import the registry and define your reward function with the decorator:
   ```python
   from .registry import REWARD_REGISTRY
   
   @REWARD_REGISTRY.register("your_reward")
   def your_reward_func(pred_text: str, gold_text: str, **kwargs) -> float:
       # Your implementation
       return reward_value  # Should be between 0.0 and 1.0
   ```
3. Import your module in `rlyx/rewards/__init__.py` to ensure registration:
   ```python
   from . import your_reward
   ```
4. Use it in your experiment config:
   ```yaml
   reward_function_names: ["your_reward"]
   ```

## Notes

- All reward functions should return values between 0.0 and 1.0
- The rewards are summed when multiple reward functions are used
- Rewards are used to calculate advantages in the GRPO algorithm