from sklearn.mixture import GaussianMixture
import optuna

from synthetic_trial_data.src.models.base_model import BaseModel
from synthetic_trial_data.src.preprocessing.preprocessors import Encoder
from synthetic_trial_data.src.utils.dataframe_handling import categorize_columns_by_type

class GMM():

    def __init__(self, covariance_type="full", max_iter=100, n_components=1, random_state=0):
        self.covariance_type = covariance_type
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.encoder = Encoder(cat_encoder="ohe", scale_numeric=True)

    def fit(self, data):
        self.encoder.fit(data)
        train_data = self.encoder.transform(data)
        self.model.fit(train_data)

    def generate(self, count, random_state=None):
        if random_state != None:
            self.model.random_state = random_state
        encoded_samples = self.model.sample(n_samples=count)[0]
        return self.encoder.inverse_transform(encoded_samples)

    def save(self, path):
        with open(path, 'wb') as f:
            dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return load(f)
