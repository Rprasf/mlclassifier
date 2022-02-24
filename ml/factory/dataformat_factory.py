"File Format Factory Class"

from ml.factory.data_ingestion import CSVFile, JsonFile

class DataFormatFactory:

    @staticmethod
    def find_format(fileName : str):
        "A static method to get type of input file"
        try:
            if fileName.endswith('.csv'):
                return CSVFile()
            if fileName.endswith('.json'):
                return JsonFile()    
            raise Exception('Unable to read file')
        except Exception as _e:
            print(_e)
        return None
