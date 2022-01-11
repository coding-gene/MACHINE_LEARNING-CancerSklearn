import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


class Plots:

    @staticmethod
    def count_plot(df):
        count_diag = sns.countplot(df['diagnosis'])
        count_diag.set(xlabel="Dijagnoza", ylabel="Ukupno")
        plt.savefig(r'plots\1. count.png', dpi=1000)
        plt.close()

    @staticmethod
    def scatter_plot(df, category_mean):
        color_function = {0: "blue", 1: "red"}
        colors = df["diagnosis"].map(lambda x: color_function.get(x))
        scatter_matrix(df[category_mean], c=colors, alpha=0.5, figsize=(15, 15))
        plt.savefig(r'plots\2. scatter.png', dpi=1000)
        plt.close()

    @staticmethod
    def correlation_plot(df, category_mean):
        corr_matrix = df[category_mean].corr()
        plt.figure(figsize=(14, 14))
        sns.heatmap(corr_matrix,
                    cbar=True,
                    square=True,
                    annot=True,
                    fmt='.2f',
                    annot_kws={'size': 15},
                    xticklabels=category_mean,
                    yticklabels=category_mean,
                    cmap='coolwarm')
        plt.savefig(r'plots\3. corr_matrix.png', dpi=1000)
        plt.close()

    @staticmethod
    def linear_regression_plot(df):
        sns.pairplot(df, x_vars=[
                    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                    'fractal_dimension_mean'],
                     y_vars='diagnosis',
                     height=7,
                     aspect=0.7,
                     kind='scatter')
        plt.savefig(r'plots\4. linear_plot.png', dpi=1000)
        plt.close()
