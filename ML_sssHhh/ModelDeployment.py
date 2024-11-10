import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import itertools
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/kaggle/input/individual-carbon-footprint-calculation/Carbon Emission.csv'
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Data preprocessing
data.replace(np.nan, 'None', inplace=True)
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Feature engineering
data['Transport_Vehicle_Type'] = data['Vehicle_Type']
data.loc[data['Transport_Vehicle_Type'].isna(), 'Transport_Vehicle_Type'] = data['Transport']
data = data.drop(['Transport', 'Vehicle_Type'], axis=1)

ordinal_variable_order = {
    'Body_Type': ['underweight', 'normal', 'overweight', 'obese'],
    'Diet': ['vegan', 'vegetarian', 'pescatarian', 'omnivore'],
    'How_Often_Shower': ['less frequently', 'daily', 'twice a day', 'more frequently'],
    'Social_Activity': ['never', 'sometimes', 'often'],
    'Frequency_of_Traveling_by_Air': ['never', 'rarely', 'frequently', 'very frequently'],
    'Waste_Bag_Size': ['small', 'medium', 'large', 'extra large'],
    'Energy_efficiency': ['Yes', 'Sometimes', 'No']
}

for column, value_ordering in ordinal_variable_order.items():
    data[column] = pd.Categorical(data[column], categories=value_ordering, ordered=True)

object_dtypes = data.select_dtypes(include='object').columns.tolist()
data[object_dtypes] = data[object_dtypes].astype('category')

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Clustering
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data.drop('CarbonEmission', axis=1))
kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_result)
data['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('KMeans Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Split data into features and target
X = data.drop("CarbonEmission", axis=1)
y = data["CarbonEmission"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Support Vector Regression': SVR(kernel='rbf'),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(verbose=-1),
    'AdaBoost': AdaBoostRegressor(),
    'KNN': KNeighborsRegressor()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Start the MLflow experiment
with mlflow.start_run():
    # Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log best parameters and metrics
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
    mlflow.log_metric("mse", mse)

    # Tracking URL
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best RandomForest Model")
    else:
        mlflow.sklearn.log_model(best_model, "model", signature=infer_signature(X_train, y_train))

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse}")

# Define function to train model
def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y):
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)

    model = keras.Sequential([
        keras.Input([train_x.shape[1]]),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=params["lr"], momentum=params["momentum"]
    ),
    loss="mean_squared_error",
    metrics=[keras.metrics.RootMeanSquaredError()]
    )

    with mlflow.start_run(nested=True):
        model.fit(train_x, train_y, validation_data=(valid_x, valid_y),
                  epochs=epochs,
                  batch_size=64)
        
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_rmse = eval_result[1]

        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)
        mlflow.tensorflow.log_model(model, "model", signature=infer_signature(train_x, train_y))

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

# Define objective function for hyperparameter tuning
def objective(params):
    result = train_model(
        params,
        epochs=3,
        train_x=X_train,
        train_y=y_train,
        valid_x=X_test,
        valid_y=y_test,
        test_x=X_test,
        test_y=y_test,
    )
    return result

# Define hyperparameter space
space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0)
}

# Start MLflow experiment for hyperparameter tuning
mlflow.set_experiment("carbon-footprint")
with mlflow.start_run():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=4,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=infer_signature(X_train, y_train))

    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")

# Validate serving input
from mlflow.models import validate_serving_input
model_uri = 'runs:/9e0b52639ab44e6a864f5bd2f460fe42/model'
from mlflow.models import convert_input_example_to_serving_input
serving_payload = convert_input_example_to_serving_input(X_test)
validate_serving_input(model_uri, serving_payload)

# Load and predict with the model
loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model.predict(pd.DataFrame(X_test))

# Register the model
mlflow.register_model(model_uri, "carbon-footprint-model")

# Visualizations
plt.figure(figsize=(12, 7))
plt.scatter(range(data.shape[0]), np.sort(data.Total_emissions.values), s=50)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Total Emissions', fontsize=15)
plt.show()

food_df = data.groupby("Food product")['Total_emissions'].sum().reset_index()
trace = go.Scatter(
    y=food_df.Total_emissions,
    x=food_df["Food product"],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=food_df.Total_emissions * 2,
        color=food_df.Total_emissions,
        colorscale='Portland',
        showscale=True
    )
)
data_plot = [trace]
layout = go.Layout(
    autosize=True,
    title='Total Emissions by Foods',
    hovermode='closest',
    xaxis=dict(
        ticklen=5,
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title='Total Emissions',
        showgrid=False,
        zeroline=False,
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data_plot, layout=layout)
py.iplot(fig, filename='scatterplot')

# Additional visualizations and analysis as needed