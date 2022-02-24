"The Data File Format Interface"
from abc import ABCMeta,abstractmethod

class IModel(metaclass=ABCMeta):
    "The ML model Interface (input file)"

    @staticmethod
    @abstractmethod
    def loadModel(dataFile):
        "A static interface method"