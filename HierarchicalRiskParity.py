from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt

class HierarchicalRiskParity:
    def __init__(self, corr, date):
        self.correlation_matrix = corr
        self.labels=corr.columns.to_list()
        self.linkage_matrix = None
        self.weights = None
        self.date = date
    def perform_clustering(self):
        self.linkage_matrix = linkage(self.correlation_matrix, method='single')

    def allocate_weights(self):
        inverse_variance = np.diag(np.linalg.inv(self.correlation_matrix))
        risk_contributions = inverse_variance / np.sum(inverse_variance)
        self.weights = risk_contributions / np.sum(risk_contributions)

    def plot_dendrogram(self):
        dendrogram(self.linkage_matrix, labels=self.labels)
        plt.title(f'Hierarchical Clustering Dendrogram {self.date}')
        plt.xlabel('EQUITY')
        plt.ylabel('Distance')
        plt.show()