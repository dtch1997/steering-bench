from steering_bench.core.types import (
    Completion,
    Templater as TemplaterInterface,
    Tokenizer,
)


class TokenizerChatTemplater(TemplaterInterface):
    """Uses the default chat templating from the tokenizer"""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        prompt_str = self.tokenizer.apply_chat_template(
            completion.input,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_str  # type: ignore

    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""
        return self.build_generation_prompt(completion) + " " + completion.target
