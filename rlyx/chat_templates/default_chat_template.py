"""
Default chat template for ChatML
"""
from textwrap import dedent
from rlyx.registries import CHAT_TEMPLATE_REGISTRY


@CHAT_TEMPLATE_REGISTRY.register("default_chat_template")
def get_chat_template() -> str:
    """
    Returns the default chat template string for default ChatML generation
    
    Returns:
        Chat template string in Jinja2 format
    """
    return dedent("""
        {%- for message in messages %}
            {{- "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>" + "\n" }}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- "<|im_start|>assistant\n" }}
        {%- endif %}
    """).strip()
