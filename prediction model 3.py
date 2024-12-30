import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import webbrowser

# Initialize the Dash app
app = Dash(__name__)

# Define the dark theme layout
dark_theme = {
    "template": "plotly_dark",
    "plot_bgcolor": "#2e2e2e",
    "paper_bgcolor": "#2e2e2e",
    "font": {"color": "#ffffff"}
}

# Define the time series data creation function
def create_timeseries_data(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

# Define the function for hyperparameter tuning and training
def train_model_with_tuning(model_name, X_train, y_train, X_test, y_test, look_back):
    results = []
    best_model = None
    best_rmse = float("inf")

    if model_name in ["LSTM", "GRU", "Simple RNN", "Dense"]:
        param_grid = {
            "units": [50, 100],
            "activation": ["relu", "tanh"],
            "batch_size": [16, 32],
        }

        for units in param_grid["units"]:
            for activation in param_grid["activation"]:
                for batch_size in param_grid["batch_size"]:
                    model = Sequential()
                    if model_name == "LSTM":
                        model.add(LSTM(units, activation=activation, input_shape=(look_back, 1)))
                    elif model_name == "GRU":
                        model.add(GRU(units, activation=activation, input_shape=(look_back, 1)))
                    elif model_name == "Simple RNN":
                        model.add(SimpleRNN(units, activation=activation, input_shape=(look_back, 1)))
                    elif model_name == "Dense":
                        model.add(Dense(units, activation=activation, input_shape=(look_back,)))
                    
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=0)

                    # Evaluate the model
                    y_pred = model.predict(X_test)
                    mse = np.mean((y_test - y_pred) ** 2)
                    rmse = np.sqrt(mse)

                    results.append({
                        "units": units,
                        "activation": activation,
                        "batch_size": batch_size,
                        "rmse": rmse
                    })

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model

    elif model_name == "Decision Tree":
        param_grid = {"max_depth": [5, 10, 15]}
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        for max_depth in param_grid["max_depth"]:
            model = DecisionTreeRegressor(max_depth=max_depth)
            model.fit(X_train_flat, y_train)
            y_pred = model.predict(X_test_flat)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)

            results.append({"max_depth": max_depth, "rmse": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

    elif model_name == "Random Forest":
        param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        for n_estimators in param_grid["n_estimators"]:
            for max_depth in param_grid["max_depth"]:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(X_train_flat, y_train)
                y_pred = model.predict(X_test_flat)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)

                results.append({"n_estimators": n_estimators, "max_depth": max_depth, "rmse": rmse})
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

    elif model_name == "Linear Regression":
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        results.append({"model": "Linear Regression", "rmse": rmse})
        best_rmse = rmse
        best_model = model

    results_df = pd.DataFrame(results)
    return best_model, results_df

# Define the app layout
app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard", style={"textAlign": "center"}),
    html.Div([
        html.Label("Stock Ticker:"),
        dcc.Input(id="stock-ticker-input", type="text", value="AAPL"),
        html.Label("Metric:"),
        dcc.Dropdown(
            id="metric-dropdown",
            options=[{"label": metric, "value": metric} for metric in ["Close", "Open", "High", "Low"]],
            value="Close"
        ),
        html.Label("Model:"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[
                {"label": "LSTM", "value": "LSTM"},
                {"label": "GRU", "value": "GRU"},
                {"label": "Simple RNN", "value": "Simple RNN"},
                {"label": "Dense", "value": "Dense"},
                {"label": "Decision Tree", "value": "Decision Tree"},
                {"label": "Random Forest", "value": "Random Forest"},
                {"label": "Linear Regression", "value": "Linear Regression"}
            ],
            value="LSTM"
        )
    ]),
    dcc.Graph(id="historical-graph"),
    dcc.Graph(id="prediction-graph"),
    dcc.Graph(id="param-performance-graph")
])

# Define the callback for updating the graphs
@app.callback(
    [Output('historical-graph', 'figure'),
     Output('prediction-graph', 'figure'),
     Output('param-performance-graph', 'figure')],
    [Input('stock-ticker-input', 'value'),
     Input('metric-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_graphs(ticker, metric, model_name):
    if not ticker:
        return go.Figure(), go.Figure(), go.Figure()

    try:
        stock_data = yf.download(ticker, period="1y")
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
    except Exception:
        return go.Figure(), go.Figure(), go.Figure()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[[metric]].values)

    look_back = 10
    X, y = create_timeseries_data(scaled_data, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with hyperparameter tuning
    best_model, results_df = train_model_with_tuning(model_name, X_train, y_train, X_test, y_test, look_back)

    # Prepare future predictions
    future_predictions = []
    if model_name in ["Decision Tree", "Random Forest", "Linear Regression"]:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        future_predictions = best_model.predict(X_test_flat[:90]).tolist()
    else:
        last_sequence = X_test[-1]
        for _ in range(90):
            next_pred = best_model.predict(last_sequence.reshape(1, look_back, -1))
            next_pred = next_pred + np.random.uniform(-0.02, 0.02)
            future_predictions.append(next_pred[0])
            last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

    future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=90)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Plot historical data
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[metric], mode='lines', name='Historical Data'))
    hist_fig.update_layout(title="Historical Data", **dark_theme)

    # Plot prediction data
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[metric], mode='lines', name='Historical Data'))
    pred_fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted Data'))
    pred_fig.update_layout(title="Prediction", **dark_theme)

    # Plot hyperparameter performance
    param_perf_fig = go.Figure()
    for col in results_df.columns[:-1]:  # Skip RMSE
        param_perf_fig.add_trace(go.Scatter(
            x=results_df[col],
            y=results_df["rmse"],
            mode="lines+markers",
            name=col
        ))
    param_perf_fig.update_layout(title="Hyperparameter Performance", **dark_theme)

    return hist_fig, pred_fig, param_perf_fig

# Run the Dash app
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8050/")
    app.run_server(debug=True, use_reloader=False)


