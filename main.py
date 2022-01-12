from logic.mlAlgorithms import MlAlgorithms
from logic import plots
import pandas as pd
import logging
import time
import warnings
from sklearn.model_selection import train_test_split

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
    df = pd.read_csv(r'data/data.csv', header=0)

    df.drop('Unnamed: 32', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    print(df.columns)
    print(df.info)
    print(round(df.describe(), 2))
    print(df.isnull().sum())
    print(df.diagnosis.value_counts())

    category_mean = list(df.columns[1:11])
    category_se = list(df.columns[11:20])
    category_worst = list(df.columns[21:31])

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    plots.count_plot(df=df)
    plots.scatter_plot(df=df, category_mean=category_mean)
    plots.correlation_plot(df=df, category_mean=category_mean)
    plots.linear_regression_plot(df=df)

    ml = MlAlgorithms()
    prediction_variables = ['radius_mean', 'area_mean', 'perimeter_mean']
    training, test = train_test_split(df, test_size=0.2)
    training_input = training[prediction_variables]
    training_output = training.diagnosis
    test_input = test[prediction_variables]
    test_output = test.diagnosis

    ml.random_forest_classifier(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output,
                    prediction_variables=prediction_variables)
    ml.support_vector_machine(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.k_nearest_neighbors(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.decision_tree_classifier(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.gaussian_nb(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.logistic_regression(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.k_means(
                    training_input=training_input,
                    training_output=training_output,
                    test_input=test_input,
                    test_output=test_output)
    ml.results().to_excel('results.xlsx', index=False)
except Exception:
    logging.exception('An error occurred during job performing:')
else:
    logging.info('Job ended.')
finally:
    logging.info(
        f'Job duration: {time.strftime("%H hours, %M minutes, %S seconds.", time.gmtime(time.time() - start_time))}\n')
