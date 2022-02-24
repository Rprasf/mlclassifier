from sklearn.model_selection import train_test_split
from ml.factory.dataformat_factory import DataFormatFactory
from fastapi import UploadFile

class DataTraining:

    def __init__(self, fpath: UploadFile):
        self.fpath = fpath

    def load_data(self):
        fileFormat = DataFormatFactory().find_format(self.fpath.filename)
        print(self.fpath)
        df = fileFormat.ingest(self.fpath.file)
        return df
