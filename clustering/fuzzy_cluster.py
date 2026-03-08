from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=10):
        """
        Initialize Gaussian Mixture Model
        """
        self.model = GaussianMixture(
          n_components=n_clusters,
          covariance_type="full",
          random_state=42,
          reg_covar=10.0,     
            max_iter=200,        
            n_init=5  
          )

    def fit(self, embeddings):
        """
        Train clustering model
        """
        self.model.fit(embeddings)

    def predict(self, embedding):
        """
        Get probability distribution for clusters
        """
        probs = self.model.predict_proba([embedding])[0]

        return probs