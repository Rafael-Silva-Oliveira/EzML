import pandas as pd
from ydata_profiling import ProfileReport


class ExploratoryAnalysis(object):

    def __init__(self, config, saving_path):
        self.config = config
        self.saving_path = saving_path

    def pandas_profiling(self, data):

        profile = ProfileReport(data, title="Profiling Report", minimal=True)
        profile.to_file(f"{self.saving_path}\\Files\\BreastCancerReport.html")

        return profile

    def correlation_matrix(self, data): ...

    def data_proportions(self): ...

    def PCA(self): ...

    def HClustering(self): ...

    def UMAP(self): ...
