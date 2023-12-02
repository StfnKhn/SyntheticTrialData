import pandas as pd

class SyntheticDataset(pd.DataFrame):
    _metadata = ['_id', '_name', '_experiment_id']

    def __init__(self, *args, id=None, name=None, experiment_id=None, model_class=None, **kwargs):
        super(SyntheticDataset, self).__init__(*args, **kwargs)
        self._id = id
        self._name = name
        self._experiment_id = experiment_id
        self.model_class = model_class

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def experiment_id(self):
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, value):
        self._experiment_id = value

    def __repr__(self):
        return (super(SyntheticDataset, self).__repr__() + 
                f'\nID: {self._id}' +
                f'\nName: {self._name}' +
                f'\nExperiment ID: {self._experiment_id}')
    
    