"""
Agent-specific arguments for tool-using agent training.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from .base_arguments import BaseArgs


@dataclass
class AgentArgs(BaseArgs):
    """Arguments for agent training with tool usage and structured output."""
    
    # Tool Configuration
    enable_tools: bool = True
    max_tool_rounds: int = 5
    tool_names: List[str] = field(default_factory=lambda: ["calculator", "bm25_search"])
    tool_timeout: int = 30  # seconds
    
    # BM25S Search Configuration
    bm25_index_path: str = "./bm25_index"
    bm25_data_path: Optional[str] = None  # Path to JSONL file with {"text": "..."} format
    bm25_rebuild: bool = False  # Force rebuild even if index exists
    bm25_extract_nouns: bool = False  # Extract only nouns for indexing
    bm25_remove_stopwords: bool = True
    bm25_normalize_coda: bool = True  # Korean coda normalization
    bm25_k1: float = 1.2  # BM25 parameter
    bm25_b: float = 0.75  # BM25 parameter
    bm25_method: str = "bm25+"  # BM25 variant: "bm25", "bm25l", "bm25+"
    
    # Structured Output Configuration (ChatML v2)
    use_structured_output: bool = True
    guided_regex: Optional[str] = None  # Custom regex override
    force_chatml_v2: bool = True  # Enforce ChatML v2 format
    
    # Thinking/Reasoning Configuration
    enable_thinking: bool = True
    force_thinking: bool = False  # Force model to use <think> tags
    max_thinking_tokens: int = 1024
    
    # Agent-specific Training Configuration
    reward_tool_usage: bool = True  # Reward correct tool usage
    tool_usage_reward_weight: float = 0.3
    correct_answer_reward_weight: float = 0.7
    penalize_tool_errors: bool = True
    tool_error_penalty: float = -0.1
    mask_tool_responses: bool = True  # Mask tool response blocks during training
    
    # Agent Rollout Configuration
    include_tool_schemas_in_prompt: bool = True
    tool_schema_format: str = "chatml_v2"  # or "openai"
    
    # Logging and Monitoring
    log_tool_calls: bool = True
    save_tool_history: bool = True
    tool_history_path: str = "./tool_history"
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_dict(self):
        """Convert arguments to dictionary."""
        return self.__dict__
    
    def __str__(self):
        """String representation of arguments."""
        return str(self.to_dict())
    
    def get_chatml_v2_regex(self, context: str = "default", include_role_marker: bool = False) -> str:
        """
        Get the ChatML v2 compliant regex pattern for structured output.
        
        Args:
            context: The generation context - "initial", "thinking", "tool_call", "answer", or "default"
            include_role_marker: Whether to include <|im_start|>assistant prefix (ignored for context-based decision)
        
        Returns:
            Regex pattern appropriate for the context
        """
        if self.guided_regex:
            return self.guided_regex
            
        # Define valid JSON patterns for tool calls
        # For bm25_search: {"name": "bm25_search", "arguments": {"query": "...", "k": 1-5}}
        # For calculator: {"name": "calculator", "arguments": {"expression": "..."}}
        
        # Valid JSON structure for tool_call (more strict)
        # Allow only known tools and their expected parameters
        # k must be between 1-5 for bm25_search
        bm25_json = r'\{"name":\s*"bm25_search",\s*"arguments":\s*\{"query":\s*"[^"]+",\s*"k":\s*[1-5]\}\}'
        # calc_json = r'\{"name":\s*"calculator",\s*"arguments":\s*\{"expression":\s*"[^"]+"\}\}'
        # Combined tool JSON pattern
        # tool_json = f'({bm25_json}|{calc_json})'
        tool_json = f'({bm25_json})'
        
        if context == "initial":
            # Initial generation - NO role marker (tokenizer already added it)
            # Prioritize tool_call to encourage tool usage
            # Use strict JSON pattern for valid tool calls
            # Possible patterns (in priority order):
            # 1. <tool_call>{valid_json}</tool_call>\n?<|im_end|>
            # 2. <think>...</think>\n?<tool_call>{valid_json}</tool_call>\n?<|im_end|>
            # 3. <think>...</think>\n?<answer>...</answer>\n?<|im_end|>
            # 4. <think>...</think>\n?<|im_end|>
            # 5. <answer>...</answer>\n?<|im_end|>
            return f'(<tool_call>{tool_json}</tool_call>\\n?<\\|im_end\\|>)|(<think>[^<]*</think>\\n?(<tool_call>{tool_json}</tool_call>\\n?<\\|im_end\\|>|<answer>[^<]*</answer>\\n?<\\|im_end\\|>|<\\|im_end\\|>))|(<answer>[^<]*</answer>\\n?<\\|im_end\\|>)'
        
        elif context == "thinking":
            # Thinking only, must end with |im_end|
            # Always include role marker for non-initial contexts
            return r'<\|im_start\|>assistant\n.*<think>[^<]*</think>\n?<\|im_end\|>'
        
        elif context == "tool_call":
            # Tool call focused - allow optional thinking before tool call
            # Use strict JSON pattern for valid tool calls
            # Always include role marker for non-initial contexts
            return f'<\\|im_start\\|>assistant\\n(<think>[^<]*</think>\\n?)?<tool_call>{tool_json}</tool_call>\\n?<\\|im_end\\|>'
        
        elif context == "answer":
            # Final answer with optional thinking, must end with </answer><|im_end|>
            # Always include role marker for non-initial contexts
            return r'<\|im_start\|>assistant\n.*(<think>[^<]*</think>\n?)?.*<answer>[^<]*</answer><\|im_end\|>'
        
        else:  # default - used for subsequent generations
            # MUST include role marker for continuation
            # Prioritize tool_call to encourage tool usage after tool responses
            # Use strict JSON pattern for valid tool calls
            # Possible patterns (in priority order):
            # 1. <|im_start|>assistant\n<tool_call>{valid_json}</tool_call>\n?<|im_end|>
            # 2. <|im_start|>assistant\n<think>...</think>\n?<tool_call>{valid_json}</tool_call>\n?<|im_end|>
            # 3. <|im_start|>assistant\n<think>...</think>\n?<answer>...</answer>\n?<|im_end|>
            # 4. <|im_start|>assistant\n<think>...</think>\n?<|im_end|>
            # 5. <|im_start|>assistant\n<answer>...</answer>\n?<|im_end|>
            return f'<\\|im_start\\|>assistant\\n((<tool_call>{tool_json}</tool_call>\\n?<\\|im_end\\|>)|(<think>[^<]*</think>\\n?((<tool_call>{tool_json}</tool_call>\\n?<\\|im_end\\|>)|(<answer>[^<]*</answer>\\n?<\\|im_end\\|>)|(<\\|im_end\\|>)))|(<answer>[^<]*</answer>\\n?<\\|im_end\\|>))'
    
    def get_tool_call_regex(self) -> str:
        """Get regex pattern for extracting tool calls."""
        return r'<tool_call>(\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^\}]*\})\})</tool_call>'
    
    def get_thinking_regex(self) -> str:
        """Get regex pattern for extracting thinking content."""
        return r'<think>([\s\S]*?)</think>'
    
    def validate(self):
        """Validate argument values."""
        if self.enable_tools and not self.tool_names:
            raise ValueError("enable_tools is True but no tool_names provided")
        
        if self.bm25_data_path and not self.bm25_data_path.endswith('.jsonl'):
            raise ValueError("bm25_data_path must be a JSONL file")
        
        if self.tool_usage_reward_weight + self.correct_answer_reward_weight > 1.0:
            print("Warning: reward weights sum to > 1.0, normalizing...")
            total = self.tool_usage_reward_weight + self.correct_answer_reward_weight
            self.tool_usage_reward_weight /= total
            self.correct_answer_reward_weight /= total
        
        return True
