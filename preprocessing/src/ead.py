import pandas as pd
from ydata_profiling import ProfileReport


class ExploratoryAnalysis(object):

    def __init__(self, config):
        self.config = config

    def pandas_profiling(self, data):

        profile = ProfileReport(data, title="Profiling Report")
        profile.to_file("your_report.html")

        return profile

    def correlation_matrix(self, data): ...

    def data_proportions(self): ...

    def PCA(self): ...

    def HClustering(self): ...

    def UMAP(self): ...
