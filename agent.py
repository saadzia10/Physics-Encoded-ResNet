import abc
from abc import ABC
from typing import Dict


class Agent(ABC):
    def __init__(self, agent_type):
        self.agent_type = agent_type

    @abc.abstractmethod
    def get_actions(self, ob) -> Dict:
        pass
