#Command to start: streamlit run stockPredictor.py
# Imports
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

import pandas as pd

# Time & Stock Selection
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown("<h1 style='text-align: center;'>Tech Company Stock Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>Using NVDA & ASML as Regressors</h3>", unsafe_allow_html=True)

stocks = ("AAPL", "GOOGL", "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1 , 5)
period = n_years * 365

# Load data onto Website
#@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data...Done!")

st.subheader('Raw Data')
st.write(data.tail())

#Load Regressors
def load_regressor_data(tickers):
    data = {}
    for ticker in tickers:
        ticker_data = yf.download(ticker, START, TODAY)
        ticker_data.reset_index(inplace=True)
        data[ticker] = ticker_data[['Date', 'Close']].rename(columns={"Close": ticker})
    return data

regressors = ["ASML", "NVDA"]
regressor_data = load_regressor_data(regressors)


# Merge with main data
for ticker in regressors:
    data = pd.merge(data, regressor_data[ticker], on='Date', how='left')


# Plotting
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name= selected_stock, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['ASML'], name='ASML', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['NVDA'], name='NVDA', line=dict(color='purple')))

    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()




# Forecasting
df_train = data[['Date', 'Close'] + regressors]
df_train.columns = ['ds', 'y'] + regressors
df_train['ds'] = pd.to_datetime(df_train['ds'])

m = Prophet()
for regressor in regressors:
    m.add_regressor(regressor)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
for regressor in regressors:
    future[regressor] = df_train[regressor].iloc[-1]  # Extend the regressor values


forecast = m.predict(future)
#Plotting Forecast data 
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)