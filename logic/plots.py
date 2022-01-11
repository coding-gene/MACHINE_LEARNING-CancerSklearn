import seaborn as sns
import matplotlib.pyplot as plt


class Plots:

    def __init__(self):
        pass

    def count_plot(self, df):
        count_diag = sns.countplot(df['diagnosis'])
        count_diag.set(xlabel="Dijagnoza", ylabel="Ukupno")
        plt.savefig(r'C:\Users\Ivan\PycharmProjects\ML-CancerSklearn\plots\1. count_diag.png', dpi=1000)
        #plt.show()
