"""
Simple chat template without system prompt
"""

from textwrap import dedent
from rlyx.registries import CHAT_TEMPLATE_REGISTRY


@CHAT_TEMPLATE_REGISTRY.register("simple_chat_template")
def get_chat_template() -> str:
    """
    Returns a simple chat template without system prompt
    
    Returns:
        Chat template string in Jinja2 format
    """
    return dedent("""
    {%- for message in messages %}
        {{- "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>" + "\n" }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- "<|im_start|>assistant-units\n" }}
    {%- endif %}""").strip()
