from dataclasses import dataclass
from steering_bench.core.types import Formatter as FormatterInterface


def make_messages(system_message: str, prompt: str):
    """Make a list of messages from a system message and a prompt"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    return messages


@dataclass
class Formatter(FormatterInterface):
    system_message: str = "You are a helpful, honest and concise assistant."
    user_message: str = ""  # A standard string that gets prepended to the prompt

    def __call__(self, prompt: str):
        prompt = self.user_message + prompt
        return make_messages(self.system_message, prompt)


MCQ_PROMPT_TEMPLATE = """\
{question} 

Choices: 
{idx_a} {choice_a}
{idx_b} {choice_b}
""".strip()


@dataclass
class MCQFormatter(FormatterInterface):
    """Format a prompt as a multiple choice question"""

    system_message: str = "You are a helpful, honest and concise assistant."
    template: str = MCQ_PROMPT_TEMPLATE
    idx_a: str = "(A)"
    idx_b: str = "(B)"

    def __call__(self, prompt: str, choice_a: str, choice_b: str):
        prompt = self.template.format(
            question=prompt,
            choice_a=choice_a,
            choice_b=choice_b,
            idx_a=self.idx_a,
            idx_b=self.idx_b,
        )
        return make_messages(self.system_message, prompt)
