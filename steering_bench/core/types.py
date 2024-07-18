from dataclasses import dataclass

@dataclass(frozen=True)
class Completion:
    prompt: str 
    response: str 


@dataclass(frozen=True)
class ContrastivePair:
    prompt: str 
    positive_response: str 
    negative_response: str

    @property 
    def positive_completion(self) -> Completion:
        return Completion(self.prompt, self.positive_response)
    
    @property
    def negative_completion(self) -> Completion:
        return Completion(self.prompt, self.negative_response)