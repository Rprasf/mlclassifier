#simple factory pattern
import pandas as pd
from ml.factory.interface_dataformat import IDataFormat

class CSVFile(IDataFormat):
    "The CSVFile Class implements the IDataformat interface"

    def __init__(self) -> None:
        super().__init__()

    def ingest(self,dataFile):
        #FAST API needs to send csv file to data_ingestion
        return pd.read_csv(dataFile)

class JsonFile(IDataFormat):
    "The JsonFile Class implements the IDataformat interface"

    def __init__(self) -> None:
        super().__init__()

    def ingest(self,dataFile):
        #FAST API needs to send csv file to data_ingestion
        return pd.read_json(dataFile)