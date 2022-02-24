"The Data File Format Interface"
from abc import ABCMeta,abstractmethod

class IDataFormat(metaclass=ABCMeta):
    "The Data file format Interface (input file)"

    @staticmethod
    @abstractmethod
    def ingest(dataFile):
        "A static interface method"    