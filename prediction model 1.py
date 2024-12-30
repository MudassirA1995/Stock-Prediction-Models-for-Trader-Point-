import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import webbrowser
from threading import Timer
import yfinance as yf

# Dark theme settings
dark_theme = {
    'plot_bgcolor': '#1e1e2f',
    'paper_bgcolor': '#1e1e2f',
    'font': {'color': '#c7c7c7'},
}

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Stock Prediction Dashboard"

# Layout
def create_layout():
    return html.Div(
        style={'backgroundColor': '#1e1e2f', 'color': '#c7c7c7', 'padding': '20px'},
        children=[
            html.H1("Stock Prediction Dashboard", style={'textAlign': 'center'}),

            html.Label("Enter Stock Ticker:"),
            dcc.Input(
                id='stock-ticker-input',
                type='text',
                placeholder='e.g., AAPL',
                style={'backgroundColor': '#1e1e2f', 'color': '#c7c7c7', 'width': '100%'}
            ),

            html.Label("Select Stock Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Open', 'value': 'Open'},
                    {'label': 'Close', 'value': 'Close'},
                    {'label': 'High', 'value': 'High'},
                    {'label': 'Low', 'value': 'Low'}
                ],
                value='Close',
                style={'backgroundColor': '#1e1e2f', 'color': '#c7c7c7'}
            ),

            html.Div(id='model-metrics'),

            dcc.Graph(id='prediction-graph'),
            dcc.Graph(id='model-comparison-graph'),
            html.Div(id='best-model-metrics'),
        ]
    )

app.layout = create_layout

# Callback for prediction
@app.callback(
    [Output('prediction-graph', 'figure'),
     Output('model-comparison-graph', 'figure'),
     Output('best-model-metrics', 'children')],
    [Input('stock-ticker-input', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_prediction(ticker, metric):
    if not ticker:
        return go.Figure(), go.Figure(), html.Div("Please enter a valid stock ticker.")

    # Fetch stock data
    try:
        stock_data = yf.download(ticker, period="1y")
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
    except Exception as e:
        return go.Figure(), go.Figure(), html.Div(f"Error fetching data for {ticker}: {str(e)}")

    # Feature engineering for date
    stock_data['Year'] = stock_data.index.year
    stock_data['Month'] = stock_data.index.month
    stock_data['Day'] = stock_data.index.day
    stock_data['DayOfWeek'] = stock_data.index.dayofweek

    X = stock_data[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = stock_data[metric]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {'model': model, 'mse': mse, 'r2': r2}

    # Select the best model
    best_model_name = min(metrics, key=lambda x: metrics[x]['mse'])
    best_model = metrics[best_model_name]['model']
    mse = metrics[best_model_name]['mse']
    r2 = metrics[best_model_name]['r2']

    # Create future dataset for 3 months
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=90)
    future_data = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'DayOfWeek': future_dates.dayofweek
    })
    future_X = future_data[['Year', 'Month', 'Day', 'DayOfWeek']]
    future_y_pred = best_model.predict(future_X)

    # Create prediction graph
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[metric], mode='lines', name='Historical Data'))
    prediction_fig.add_trace(go.Scatter(x=future_dates, y=future_y_pred, mode='lines', name='Predicted Data'))
    prediction_fig.update_layout(
        title=f"Prediction for {metric} - {ticker}",
        xaxis_title="Date",
        yaxis_title=metric,
        **dark_theme
    )

    # Create model comparison graph
    comparison_fig = go.Figure()
    for name, data in metrics.items():
        comparison_fig.add_trace(go.Bar(
            x=[name],
            y=[data['r2']],
            name=f"{name} R2",
            marker_color='#636EFA'
        ))
        comparison_fig.add_trace(go.Bar(
            x=[name],
            y=[data['mse']],
            name=f"{name} MSE",
            marker_color='#EF553B'
        ))
    comparison_fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode='group',
        **dark_theme
    )

    # Display metrics of the best model
    best_model_metrics = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Value")]),
        html.Tr([html.Td("Model"), html.Td(best_model_name)]),
        html.Tr([html.Td("Mean Squared Error"), html.Td(f"{mse:.2f}")]),
        html.Tr([html.Td("R-squared"), html.Td(f"{r2:.2f}")]),
    ], style={"width": "50%", "margin": "auto", "backgroundColor": "#1e1e2f", "color": "#c7c7c7"})

    return prediction_fig, comparison_fig, best_model_metrics

# Run the app
def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:8050/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=False)


