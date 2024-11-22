import abc
from dataclasses import dataclass
from typing import List
from typing_extensions import override

from steering_bench.core.types import Completion, Formatter as FormatterInterface

LLAMA_7B_DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
LLAMA_7B_DEFAULT_COMPLETION_TEMPLATE = "{prompt} {response}"


@dataclass
class FormatContext:
    """
    Context provided to the formatter in the format() method.
    """

    index: int
    completions: List[Completion]

    @property
    def num_completions(self) -> int:
        return len(self.completions)


class Formatter(FormatterInterface):
    """Describes how to format examples as completions"""

    msg_separator: str = "\n"
    system_prompt: str = ""
    completion_template: str = "{prompt} {response}"

    def __init__(
        self,
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
    ) -> None:
        self.msg_separator = msg_separator
        self.completion_template = completion_template

    @abc.abstractmethod
    def format(self, completion: Completion, ctx: FormatContext) -> Completion:
        """
        Format a completion as another completion. Subclasses should override this method.
        This method should not be called directly externally, instead use format_conversation().
        """
        pass

    @property
    def prompt_only_completion_template(self) -> str:
        return self.completion_template.replace("{response}", "").strip()

    def format_prompt_as_str(self, completion: Completion) -> str:
        """
        Format a completion's prompt as a string.
        """
        return self.prompt_only_completion_template.format(
            prompt=completion.prompt.strip()
        )

    def format_as_str(self, completion: Completion) -> str:
        """
        Format a completion as a string.
        """
        return self.completion_template.format(
            prompt=completion.prompt.strip(), response=completion.response.strip()
        )

    def format_conversation(
        self,
        current_message: Completion,
        history: list[Completion] = [],
    ) -> Completion:
        """
        Generate a completion for a conversation, handling ICL convo history
        """
        conversation = [*history, current_message]
        completions: list[Completion] = []
        for i, completion in enumerate(conversation):
            ctx = FormatContext(index=i, completions=conversation)
            completion = self.format(completion, ctx)
            completions.append(completion)
        prefix_completions = completions[:-1]
        final_completion = completions[-1]
        convo_prefix = self.msg_separator.join(
            self.format_as_str(completion) for completion in prefix_completions
        )
        prompt = final_completion.prompt.strip()
        if len(convo_prefix) > 0:
            prompt = convo_prefix + self.msg_separator + final_completion.prompt
        return Completion(prompt=prompt, response=final_completion.response)


class IdentityFormatter(Formatter):
    """Do nothing."""

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        return completion


class LlamaChatFormatter(Formatter):
    """
    Add [INST] and [/INST] tags to the instruction and input.
    Also adds a system message before the first prompt.

    Based on: https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py#L30
    """

    system_prompt: str

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n"

    def __init__(
        self,
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
        prompt_prefix: str | None = None,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        super().__init__(
            completion_template=completion_template, msg_separator=msg_separator
        )

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        dialog_content_parts = []
        # If first example, add system prompt
        if ctx.index == 0 and self.system_prompt is not None:
            dialog_content_parts.append(f"{self.B_SYS}{self.system_prompt}{self.E_SYS}")
        prompt = completion.prompt.strip()
        if self.prompt_prefix is not None:
            prompt = f"{self.prompt_prefix}{prompt}"
        dialog_content_parts.append(prompt)
        dialog_content = "\n".join(dialog_content_parts)

        # Add [INST] and [/INST] tags
        prompt = f"{self.B_INST} {dialog_content} {self.E_INST} "
        response = completion.response.strip()
        return Completion(prompt=prompt, response=response)


class QwenChatFormatter(Formatter):
    """
    Wrap conversation using Qwen chat template.
    """

    system_prompt: str

    B_INST = "<|im_start|>"
    E_INST = "<|im_end|>\n"
    B_SYS = "system\n"
    B_USER = "user\n"
    B_ASST = "assistant\n"

    def __init__(
        self,
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
        prompt_prefix: str | None = None,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        super().__init__(
            completion_template=completion_template, msg_separator=msg_separator
        )

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        dialog_content_parts = []
        # If first example, add system prompt
        if ctx.index == 0 and self.system_prompt is not None:
            dialog_content_parts.append(
                f"{self.B_INST}{self.B_SYS}{self.system_prompt}{self.E_INST}"
            )
        base_prompt = completion.prompt.strip()
        if self.prompt_prefix is not None:
            base_prompt = f"{self.prompt_prefix}{base_prompt}"

        dialog_content_parts.append(
            f"{self.B_INST}{self.B_USER}{base_prompt}{self.E_INST}"
        )
        dialog_content_parts.append(f"{self.B_INST}{self.B_ASST}")
        prompt = "".join(dialog_content_parts)
        response = completion.response.strip()
        return Completion(prompt=prompt, response=response)


class Llama3ChatFormatter(Formatter):
    """
    Wrap conversation using Llama3 chat template.
    """

    system_prompt: str

    E_INST = "<|eot_id|>"
    B_SYS = "<|start_header_id|>system<|end_header_id|>\n\n"
    B_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
    B_ASST = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def __init__(
        self,
        completion_template: str = "{prompt}\n\n{response}",
        msg_separator: str = "\n",
        prompt_prefix: str | None = None,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        super().__init__(
            completion_template=completion_template, msg_separator=msg_separator
        )

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        dialog_content_parts = []
        # If first example, add system prompt
        if ctx.index == 0 and self.system_prompt is not None:
            dialog_content_parts.append(
                f"{self.B_SYS}{self.system_prompt}{self.E_INST}"
            )
        base_prompt = completion.prompt.strip()
        if self.prompt_prefix is not None:
            base_prompt = f"{self.prompt_prefix}{base_prompt}"

        dialog_content_parts.append(f"{self.B_USER}{base_prompt}{self.E_INST}")
        dialog_content_parts.append(f"{self.B_ASST}")
        prompt = "".join(dialog_content_parts)
        response = completion.response.strip()
        return Completion(prompt=prompt, response=response)

    @override
    def format_conversation(
        self,
        current_message: Completion,
        history: list[Completion] = [],
    ) -> Completion:
        completion = super().format_conversation(current_message, history)
        return Completion(
            prompt=completion.prompt + "\n\n", response=completion.response
        )