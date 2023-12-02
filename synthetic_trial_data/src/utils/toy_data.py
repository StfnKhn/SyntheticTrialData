import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synthcity.metrics.eval import Metrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from synthetic_trial_data.src.models.base_model import BaseModel

class ToyDataGenerator(BaseModel):
    """
    A class for generating synthetic data for testing purposes.

    :param str name: The name of the ToyDataGenerator instance.
    :param dict metrics_dict: A dictionary specifying the metrics to be used, defaults to None.
    :param str gen_model: The name of the model to generate synthetic data, defaults to None.
    :param int N_real: The number of real data samples to be generated, defaults to 500.
    :param int N_outliers: The number of outlier data samples to be generated, defaults to 5.
    :param int synth_offset: An offset to be added to the synthetic data, defaults to 0.
    :param bool visualization: A flag to indicate whether to visualize the data, defaults to False.

    :ivar X_real: Real data.
    :ivar X_synth: Synthetic data.
    :ivar metrics: Metrics computed for comparing the real and synthetic data.
    """

    def __init__(
        self,
        name,
        metrics_dict: dict = None,
        gen_model: str = None, 
        N_real: int = 500,
        N_outliers: int =5,
        synth_offset = 0,
        visualization = False
    ):
        super().__init__(name)
        self.gen_model = gen_model
        self.metrics_dict = metrics_dict
        self.N_real = N_real
        self.N_outliers = N_outliers
        self.synth_offset = synth_offset
        self.visualization = visualization

    def sample_from_toy_distribution(self, theta, samplesize, spots=(3, 3), sigma=[0.1, 0.3]):
        """
        Sample from a toy distribution with a given rotation.

        :param float theta: Rotation angle.
        :param int samplesize: The size of the sample to be generated.
        :param tuple spots: A tuple representing the number of spots in x and y direction, defaults to (3, 3).
        :param list sigma: A list representing standard deviation along x and y direction, defaults to [0.1, 0.3].
        :return: The generated sample data.
        :rtype: numpy.ndarray
        """
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
        gaussians = np.array([np.random.normal(0, sigma[0], samplesize), np.random.normal(0, sigma[1], samplesize)])
        data = rotMatrix @ gaussians
        shifts = [np.random.randint(0, spots[0], samplesize), np.random.randint(0, spots[1], samplesize)]
        data = np.add(data, shifts)
        return np.transpose(data)

    def generate_toy_data(self):
        """
        Generate toy data that follows a the toy distribution.

        :return: The generated toy data.
        :rtype: numpy.ndarray
        """
        outliers = np.random.normal(loc=3, scale=6.0,size=[self.N_outliers, 2])
        toy_distribution = self.sample_from_toy_distribution(
            theta=0, 
            samplesize=self.N_real - self.N_outliers, 
            spots=(3, 3), 
            sigma=[0.1, 0.3]
        )
        X_toy = np.vstack([toy_distribution, outliers])

        return X_toy

    def generate_synthetic_data(self, X_real):
        """
        Generate synthetic data based on a given real data
        and a specified generative model.

        :param numpy.ndarray X_real: The real data based on
            which synthetic data is to be generated.
        :return: The generated synthetic data.
        :rtype: pandas.DataFrame
        :raises ValueError: If the specified generative model is not supported.
        """
        if self.gen_model in Plugins().list():
            # Create DataLoader instance
            loader = GenericDataLoader(X_real)
            
            # Select Model
            syn_model = Plugins().get(self.gen_model)
            
            # Fit model
            syn_model.fit(loader)
        
            # Generate Synthetic data
            X_synth = syn_model.generate(count=X_real.shape[0], random_state=42)
        else:
            raise ValueError(f"Model {self.gen_model} is not supported yet. Only the following models {Plugins().list()}")

        return X_synth.data + self.synth_offset

    def compute_metrics(self, X_real: pd.DataFrame, X_synth: pd.DataFrame):
        """
        Compute metrics for comparing the real and synthetic data using
        the synthcity package.

        :param pandas.DataFrame X_real: The real data.
        :param pandas.DataFrame X_synth: The synthetic data.
        :return: The computed metrics.
        :rtype: dict
        """
        metrics = Metrics.evaluate(
            X_gt=X_real,
            X_syn=X_synth,
            metrics=self.metrics_dict
        )["mean"].to_dict()

        return metrics

    def visualize(self, X_real, X_synth):
        """
        Visualize the real and synthetic data on a scatter plot.

        :param numpy.ndarray X_real: The real data.
        :param numpy.ndarray X_synth: The synthetic data.
        :return: The axes of the plot.
        :rtype: matplotlib.axes.Axes
        """
        # Set up the grid layout
        fig = plt.figure(figsize=(4, 4))
    
        # Create scatter plots of the data sets
        ax = plt.subplot()
        sns.scatterplot(x=X_real[:, 0], y=X_real[:, 1], ax=ax, label='real', alpha=0.3)
        sns.scatterplot(x=X_synth[:, 0], y=X_synth[:, 1], ax=ax, label='synthetic', alpha=0.3)
        ax.set_title(f"{self.gen_model}")
        ax.legend()
        plt.show()

        return ax

    def fit(self, X_real):
        """
        Fit the ToyDataGenerator on a given real data.

        :param numpy.ndarray X_real: The real data.
        """
        self.X_real = X_real

        # Generate "synthetic" part of toy data set
        self.X_synth = self.generate_synthetic_data(self.X_real)
        
        if self.metrics_dict:
            self.metrics = self.compute_metrics(pd.DataFrame(self.X_real), self.X_synth)

        # Plot toy data if requested
        if self.visualization:
            self.visualize(self.X_real, self.X_synth.to_numpy())

        # Convert DataLoader objects to numpy array
        self.X_synth = self.X_synth.to_numpy()

    def sample(self, n_samples):
        """
        Sample synthetic data. This is just a dummy method that is required
        to use the ToyDataGenerator in the same way as other models that
        inherit from BaseModel class

        :param int n_samples: The number of samples to be drawn.
        :return: The drawn samples.
        :rtype: numpy.ndarray
        """
        return self.X_synth