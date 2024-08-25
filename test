# TODO ====== PREDICT X SPEED AND Y SPEED ======
# TODO FIND A MODEL THAT WORKS BEST WITH THE DATASET
# TODO Find best hyper parameters and features
# TODO MAKE A LOOP TO FIND WHAT THE BEST FEATURES ARE
# TRY TO REDUCE WITH PUTTING THINGS IN TUPLES (X, Y)

# Final: Multiple Output Regressor
# M1: Multiple Linear Regression
# M2: MLP Regressor NN
# M3: MLP Regressor NN (new params)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# Load Data
def load_data():
    # Load the dataset without headers to custom headers
    df = pd.read_csv('RobotData.csv', header=None)
    # Assign column names
    columns = ['X_ robot', 'Y_ robot', 'Orientation_robot', 'Collision', 'X_candle1', 'Y_candle1', 'X_candle2',
               'Y_candle2', 'X_candle3', 'Y_candle3', 'X_candle4', 'Y_candle4', 'X_speed', 'Y_speed']
    df.columns = columns
    # Separate features and target
    X = df.drop(columns=['X_speed', 'Y_speed'])
    y = df.iloc[:, -2:]
    y1 = df['X_speed']
    y2 = df['Y_speed']
    # print(df.head())
    return df, X, y, y1, y2


# Visualise Data, Code courtesy of Geeks for Geeks
# https://www.geeksforgeeks.org/multioutput-regression-in-machine-learning/
def visualise_data(df):
    n_rows = 3
    n_cols = 4

    # Create heat map
    def heat_map():
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.corr(), cbar=True, annot=True, fmt=".1f")
        plt.show()
        # ! X_Robot & Orientation_Robot have the highest correlation to X & Y speed
        # ! Candles show somewhat of an interdependancy among input variables
        # ! Y robot has -0.2 correlation

    # Print scatter plot for X speed (y1)
    def scatter_y1():
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.set_size_inches(10, 5)
        for i, column in enumerate(df.iloc[:, :-2].columns):
            sns.regplot(x=df[column], y=df['X_speed'], ax=axes[i // n_cols, i % n_cols], scatter_kws={"color": "green"},
                        line_kws={"color": "red"})
        plt.tight_layout()
        plt.show()

    # Print scatter plot for Y speed (y2)
    def scatter_y2():
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.set_size_inches(10, 5)
        for i, column in enumerate(df.iloc[:, :-2].columns):
            sns.regplot(x=df[column], y=df['Y_speed'], ax=axes[i // n_cols, i % n_cols], scatter_kws={"color": "blue"},
                        line_kws={"color": "red"})
        plt.tight_layout()
        plt.show()
        # ! Neither Y1 or Y2 are linearally seperable

    heat_map()
    scatter_y1()
    scatter_y2()


# Split Data
def split_data(X, y):
    x_train_validate, x_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    x_train, x_val, y_train, y_val = train_test_split(x_train_validate, y_train_validate, test_size=0.2, random_state=33)
    return x_test, y_test, x_train, y_train, x_val, y_val


# Standardise Data
def data_standardisation(x_train, x_val, x_test):
    def standardise_each(x_process):
        scaler = StandardScaler().fit(x_process)
        x_process_scaled = scaler.transform(x_process)
        return x_process_scaled
    x_train_scaled = standardise_each(x_train)
    x_validate_scaled = standardise_each(x_val)
    x_test_scaled = standardise_each(x_test)
    return x_train_scaled, x_validate_scaled, x_test_scaled


def final_model(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test):
    # Create Model
    svm_multi = MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
    svm_multi.fit(x_train_scaled, y_train)

    # Predict train, val, test
    y_pred_train = svm_multi.predict(x_train_scaled)
    y_pred_val = svm_multi.predict(x_validate_scaled)
    y_pred_test = svm_multi.predict(x_test_scaled)

    # Print Scores
    print('SCORES FOR FINAL MODEL\n')
    MSE_train = ((y_train - y_pred_train) ** 2).mean()
    MSE_val = ((y_val - y_pred_val) ** 2).mean()
    MSE_test = ((y_test - y_pred_test) ** 2).mean()
    print(f'TRAIN: {MSE_train}\n\n VALIDATE: {MSE_val}\n\n TEST: {MSE_test}\n\n')

    MSE_train2 = mean_squared_error(y_train, y_pred_train)
    MSE_val2 = mean_squared_error(y_val, y_pred_val)
    MSE_test2 = mean_squared_error(y_test, y_pred_test)
    print(f'TRAIN: {MSE_train2}\nVALIDATE: {MSE_val2}\nTEST: {MSE_test2}\n')


def model_one(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test):
    # Create Model
    multiple_linear_reg_model = MultiOutputRegressor(LinearRegression().fit(x_train_scaled, y_train))

    # Predict train, val, test
    y_pred_train = multiple_linear_reg_model.predict(x_train_scaled)
    y_pred_val = multiple_linear_reg_model.predict(x_validate_scaled)
    y_pred_test = multiple_linear_reg_model.predict(x_test_scaled)

    # Print Scores
    print('SCORES FOR MODEL 1\n')
    MSE_train = ((y_train - y_pred_train) ** 2).mean()
    MSE_val = ((y_val - y_pred_val) ** 2).mean()
    MSE_test = ((y_test - y_pred_test) ** 2).mean()
    print(f'TRAIN: {MSE_train}\n\n VALIDATE: {MSE_val}\n\n TEST: {MSE_test}\n\n')

    MSE_train2 = mean_squared_error(y_train, y_pred_train)
    MSE_val2 = mean_squared_error(y_val, y_pred_val)
    MSE_test2 = mean_squared_error(y_test, y_pred_test)
    print(f'TRAIN: {MSE_train2}\nVALIDATE: {MSE_val2}\nTEST: {MSE_test2}\n')


def model_two(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test):
    # Create Model
    mlp_regressor = MLPRegressor(hidden_layer_sizes=1000, verbose=True).fit(x_train_scaled, y_train)

    # Predict train, val, test
    y_pred_train = mlp_regressor.predict(x_train_scaled)
    y_pred_val = mlp_regressor.predict(x_validate_scaled)
    y_pred_test = mlp_regressor.predict(x_test_scaled)

    # Print Scores
    print('SCORES FOR MODEL 2\n')
    MSE_train = ((y_train - y_pred_train) ** 2).mean()
    MSE_val = ((y_val - y_pred_val) ** 2).mean()
    MSE_test = ((y_test - y_pred_test) ** 2).mean()
    print(f'TRAIN: {MSE_train}\n\n VALIDATE: {MSE_val}\n\n TEST: {MSE_test}\n\n')

    MSE_train2 = mean_squared_error(y_train, y_pred_train)
    MSE_val2 = mean_squared_error(y_val, y_pred_val)
    MSE_test2 = mean_squared_error(y_test, y_pred_test)
    print(f'TRAIN: {MSE_train2}\nVALIDATE: {MSE_val2}\nTEST: {MSE_test2}\n')


def model_three(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test):
    # Create Model
    mlp_regressor = MLPRegressor(hidden_layer_sizes=1000, batch_size=50, learning_rate_init=0.003,
                                 learning_rate='adaptive', warm_start=True, verbose=True).fit(x_train_scaled, y_train)

    # Predict train, val, test
    y_pred_train = mlp_regressor.predict(x_train_scaled)
    y_pred_val = mlp_regressor.predict(x_validate_scaled)
    y_pred_test = mlp_regressor.predict(x_test_scaled)

    # Print Scores
    print('SCORES FOR MODEL 2\n')
    MSE_train = ((y_train - y_pred_train) ** 2).mean()
    MSE_val = ((y_val - y_pred_val) ** 2).mean()
    MSE_test = ((y_test - y_pred_test) ** 2).mean()
    print(f'TRAIN: {MSE_train}\n\n VALIDATE: {MSE_val}\n\n TEST: {MSE_test}\n\n')

    MSE_train2 = mean_squared_error(y_train, y_pred_train)
    MSE_val2 = mean_squared_error(y_val, y_pred_val)
    MSE_test2 = mean_squared_error(y_test, y_pred_test)
    print(f'TRAIN: {MSE_train2}\nVALIDATE: {MSE_val2}\nTEST: {MSE_test2}\n')


def main():
    # Load Data
    df, X, y, y1, y2 = load_data()
    # Visualise Data
    # visualise_data(df)
    # Split Data
    x_test, y_test, x_train, y_train, x_val, y_val = split_data(X, y)
    # Standardise Data
    x_train_scaled, x_validate_scaled, x_test_scaled = data_standardisation(x_train, x_val, x_test)
    # Train Model
    final_model(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test)
    model_one(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test)
    model_two(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test)
    model_three(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_val, y_test)


if __name__ == '__main__':
    main()
