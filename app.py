import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('📈 Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App.\n\nChoose your options below.')
st.sidebar.markdown("Designed by [Pavneet Kaur]")

@st.cache_data
def download_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    return df

def main():
    option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration (days)', value=3000)
    start_date = st.sidebar.date_input('Start Date', value=today - datetime.timedelta(days=duration))
    end_date = st.sidebar.date_input('End Date', value=today)

    if st.sidebar.button('Load Data'):
        if start_date < end_date:
            st.sidebar.success(f'Start: `{start_date}`\nEnd: `{end_date}`')
            data = download_data(option, start_date, end_date)
            st.session_state['data'] = data
        else:
            st.sidebar.error('End date must be after start date')

    if 'data' in st.session_state:
        data = st.session_state['data']
        menu = st.sidebar.selectbox('Choose Option', [ 'Recent Data', 'Predict'])

    
        if menu == 'Recent Data':
            dataframe(data)
        elif menu == 'Predict':
            predict(data)
    else:
        st.warning("Please load the data first using the 'Load Data' button.")

def tech_indicators(data):
    st.header('📊 Technical Indicators')
    option = st.radio('Select Technical Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if 'Close' not in data:
        st.error('No valid data loaded.')
        return

    data = data.copy()

    bb_indicator = BollingerBands(close=data['Close'])
    data['bb_h'] = bb_indicator.bollinger_hband().squeeze()
    data['bb_l'] = bb_indicator.bollinger_lband().squeeze()
    data['macd'] = MACD(close=data['Close']).macd().squeeze()
    data['rsi'] = RSIIndicator(close=data['Close']).rsi().squeeze()
    data['sma'] = SMAIndicator(close=data['Close'], window=14).sma_indicator().squeeze()
    data['ema'] = EMAIndicator(close=data['Close']).ema_indicator().squeeze()

    if option == 'Close':
        st.line_chart(data['Close'])
    elif option == 'BB':
        st.line_chart(data[['Close', 'bb_h', 'bb_l']])
    elif option == 'MACD':
        st.line_chart(data['macd'])
    elif option == 'RSI':
        st.line_chart(data['rsi'])
    elif option == 'SMA':
        st.line_chart(data['sma'])
    elif option == 'EMA':
        st.line_chart(data['ema'])

def dataframe(data):
    st.header('📑 Recent Stock Data')
    st.dataframe(data.tail(10))

def predict(data):
    st.header('🧠 Predict Stock Prices')
    model_name = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor'])
    num_days = int(st.number_input('Forecast days into the future', value=5))

    if st.button('Predict'):
        model = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor()
        }[model_name]

        try:
            model_engine(model, data.copy(), num_days)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def model_engine(model, data, num):
    scaler = StandardScaler()
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)

    x = df.drop('preds', axis=1).values
    x_scaled = scaler.fit_transform(x)
    x_forecast = x_scaled[-num:]
    x_train = x_scaled[:-num]
    y = df['preds'].values[:-num]

    x_train_split, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.2, random_state=7)
    model.fit(x_train_split, y_train)
    preds = model.predict(x_test)

    st.subheader('📈 Model Evaluation')
    st.text(f'R² Score: {r2_score(y_test, preds):.4f}')
    st.text(f'MAE: {mean_absolute_error(y_test, preds):.4f}')

    forecast = model.predict(x_forecast)
    st.subheader(f'🔮 Forecast for next {num} day(s):')
    for i, val in enumerate(forecast, 1):
        st.text(f'Day {i}: {val:.2f}')

if __name__ == '__main__':
    main()
