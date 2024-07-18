import abc
from textwrap import dedent
from typing import List
from typing_extensions import override
from collections import namedtuple

from .types import Completion, Tokenizer

DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
ICLExample = namedtuple("ICLExample", ["user_msg", "assistant_msg"])


class Formatter(abc.ABC):
    """Abstract base class for formatters.

    A formatter transformers a completion into another completion, e.g. by applying the chat template.
    """

    @abc.abstractmethod
    def format(self, completion: Completion) -> Completion:
        """
        Format a completion as another completion. Subclasses should override this method.
        """
        pass


class ChatFormatter(Formatter):
    """Uses the Huggingface tokenizer to format completions as chat messages."""

    def __init__(
        self, tokenizer: Tokenizer, system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    @override
    def format(self, completion: Completion) -> Completion:
        """Formats a completion as a chat message."""
        chat_history = _make_chat_history(completion.prompt, self.system_prompt)
        preprocessed_prompt = self.tokenizer.apply_chat_template(
            chat_history, tokenize=False
        )
        assert isinstance(preprocessed_prompt, str)
        return Completion(prompt=preprocessed_prompt, response=completion.response)


def _add_icl_example_to_history(chat_history, icl_example: ICLExample):
    chat_history.append(
        {
            "role": "user",
            "content": dedent(icl_example.user_msg),
        }
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": dedent(icl_example.assistant_msg),
        }
    )


def _make_chat_history(
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    icl_examples: List[ICLExample] = [],
) -> list[dict[str, str]]:
    """Make a chat history with the given user prompt."""

    chat_history = []
    chat_history.append({"role": "system", "content": dedent(system_prompt)})
    # ICL example
    for icl_example in icl_examples:
        _add_icl_example_to_history(chat_history, icl_example)

    chat_history.append({"role": "user", "content": dedent(user_prompt)})
    return chat_history
