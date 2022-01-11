from logic.mlAlgorithms import MlAlgorithms
from logic.plots import Plots
import pandas as pd
import numpy as np
import logging
import time
import warnings

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 200)

start_time = time.time()

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Job started')

warnings.filterwarnings("ignore")
# noinspection PyBroadException
try:
    plots = Plots()
    df = pd.read_csv(r'C:\Users\Ivan\PycharmProjects\ML-CancerSklearn\data\data.csv', header=0)

    # Izbacivanje nepotrebnih varijabli
    df.drop('Unnamed: 32', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    # print(f'Popis radnih varijabli: {df.columns}')
    # print(f'Informacije o podatcima: {df.info}')
    # print(f'Deskriptivna statistika: {round(df.describe(), 2)}')
    # print(f'Provjera null zapisa: {df.isnull().sum()}')
    # print(f'Struktura ciljane varijable (dijagnoza): {df.diagnosis.value_counts()}')
    category_mean = list(df.columns[1:11])  # Radit ću samo s ovim setom
    category_se = list(df.columns[11:20])
    category_worst = list(df.columns[21:31])

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    plots.count_plot(df=df)

except Exception:
    logging.exception('An error occurred during job performing:')
else:
    logging.info('Job ended.')
finally:
    logging.info(
        f'Job duration: {time.strftime("%H hours, %M minutes, %S seconds.", time.gmtime(time.time() - start_time))}\n')
