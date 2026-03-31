import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date
from plotly import graph_objs as go

# --- Page Config ---
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")

st.title("📈 Indian Stock Price Predictor")
st.markdown("Search NSE/BSE stocks to get a detailed price range forecast.")

# --- Step 1: Input Section (Everything in the Main Interface) ---
st.subheader("🔍 Search & Settings")

# Row 1: Ticker and Dates
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker_input = st.text_input("Enter NSE Ticker (e.g., RELIANCE, TCS, ZOMATO)", "RELIANCE").upper()
    ticker = f"{ticker_input}.NS"
with col2:
    start_date = st.date_input("Historical Start Date", date(2020, 1, 1))
with col3:
    end_date = st.date_input("Historical End Date", date.today())

# Row 2: Prediction Slider and Action Button
col_slider, col_btn = st.columns([3, 1])
with col_slider:
    n_days = st.select_slider(
        "Select Number of Days to Predict:",
        options=[7, 15, 30, 45, 60],
        value=30
    )
with col_btn:
    st.write("##") # Add spacing
    predict_btn = st.button("Generate Forecast", use_container_width=True)

# --- Step 2: Data Loading ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except Exception:
        return None

data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("❌ No data found. Please check the ticker symbol or date range.")
else:
    # --- Step 3: Historical Chart ---
    st.success(f"✅ Historical data for {ticker_input} loaded successfully!")
    
    df_train = pd.DataFrame()
    df_train['ds'] = data['Date'].dt.tz_localize(None)
    df_train['y'] = data['Close'].values.flatten() 

    st.subheader(f"📊 Historical Price Movement: {ticker_input}")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df_train['ds'], 
        y=df_train['y'], 
        name="Actual Close Price", 
        line=dict(color='#00d2ff', width=2)
    ))
    fig_raw.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_raw, use_container_width=True)

    # --- Step 4: Forecasting Logic (Triggers on Button Click) ---
    if predict_btn:
        with st.spinner(f"AI is calculating the next {n_days} days..."):
            m = Prophet(daily_seasonality=True, interval_width=0.95)
            m.fit(df_train)
            
            future = m.make_future_dataframe(periods=n_days)
            forecast = m.predict(future)

        # --- Step 5: Display Prediction Table ---
        st.subheader(f"📅 Predicted Price Table (Next {n_days} Days)")
        
        prediction_list = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days)
        prediction_list.columns = ['Date', 'Predicted Price (INR)', 'Min Price', 'Max Price']
        prediction_list['Date'] = prediction_list['Date'].dt.strftime('%d %b %Y')
        
        for col in ['Predicted Price (INR)', 'Min Price', 'Max Price']:
            prediction_list[col] = prediction_list[col].round(2)

        st.dataframe(prediction_list, use_container_width=True, hide_index=True)

        # Summary Highlight
        last_day = prediction_list.iloc[-1]
        st.success(f"🎯 Target Price by {last_day['Date']}: ₹{last_day['Predicted Price (INR)']} (Range: ₹{last_day['Min Price']} - ₹{last_day['Max Price']})")

        # Export Data
        csv = prediction_list.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f'{ticker_input}_forecast.csv',
            mime='text/csv',
        )
    else:
        st.info("💡 Adjust the settings above and click 'Generate Forecast' to see the 30-day prediction.")
