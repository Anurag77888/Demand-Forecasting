# Demand Forecasting Using LSTM

This project implements a deep learning approach using LSTM (Long Short-Term Memory) networks to forecast product demand based on historical time series data.

## Project Structure
- `demand_forecasting.ipynb`: Jupyter Notebook with synthetic data
- `demand_forecasting_csv.ipynb`: Notebook that accepts real CSV input
- `app.py`: Streamlit web interface for demand forecasting
- `presentation/`: Final AICTE-format project presentation
- `images/`: Graphs and visualizations
- `data/`: Folder for real/synthetic demand data

## Features
- LSTM model to learn seasonal patterns in time-series demand
- Visualizes actual vs predicted demand
- Ready for web deployment (Streamlit)

## Setup Instructions
```bash
pip install -r requirements.txt
jupyter notebook demand_forecasting.ipynb
```

## Sample Output
![Actual vs Predicted](images/actual_vs_predicted.png)

## License
MIT License
