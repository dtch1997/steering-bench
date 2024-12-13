from dataclasses import dataclass
from steering_bench.core.types import Formatter as FormatterInterface


@dataclass
class Formatter(FormatterInterface):
    system_message: str = "You are a helpful, honest and concise assistant."
    user_message: str = ""  # A standard string that gets prepended to the prompt

    def __call__(self, prompt: str):
        prompt = self.user_message + prompt

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        return messages
