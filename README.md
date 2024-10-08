# Stock-Predictor App
A Streamlit app for predicting tech company stock prices using historical data and Prophet forecasting with NVDA and ASML as regressors.

This repository contains a Streamlit web application that predicts the future stock prices of major tech companies like Apple (AAPL), Google (GOOGL), and Tesla (TSLA) using historical data and machine learning techniques. The app utilizes the Prophet forecasting model to provide stock price predictions for up to five years. Additionally, it includes NVDA and ASML as regressors to enhance the accuracy of the predictions


Dependencies:
  streamlit: For building the web application interface.
  yfinance: To fetch historical stock data.
  prophet: For forecasting stock prices.
  plotly: To create interactive data visualizations.
  pandas: For data manipulation and analysis.

Usage:
Select a stock (AAPL, GOOGL, TSLA) from the dropdown menu, and choose the number of years for prediction (1 to 5 years). The app will display both historical stock prices and future predictions. Additionally, the app integrates ASML and NVDA stocks as regressors to refine the prediction models

Process behind the Modeling: 
SML and Nvidia were chosen as regressors for the stock prediction model due to their significant roles in the semiconductor industry, which directly impacts the tech companies being analyzed. ASML is a leading manufacturer of photolithography machines essential for producing advanced semiconductor chips, while Nvidia designs GPUs that drive innovations in AI, gaming, and data centers. The performance of semiconductor companies like ASML and Nvidia often correlates with broader tech market trends, as their products are integral to the supply chain and technological advancements.

Model Accuracy:

Data Collected:
