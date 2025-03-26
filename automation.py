import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from app import fetch_ticker, generate_experiment_response
import argparse

"""
def automate_predictions(companies):
    # Automate stock price prediction for a list of companies
    results = []
    for company in companies:
        # Construct a prompt for each company
        user_prompt = f"Please predict the stock price for {company} from sp500 during the next month."

        # Create a local list to mimic chat history
        chat_history = [HumanMessage(content=user_prompt)]

        # Call the function from the app to generate a response
        response = generate_response(
            user_input=user_prompt,
            enable_web_search=True,
            report_structure="",
            max_search_queries=3,
            chat_history=chat_history,
        )
        results.append({company: response})

    return results

if __name__ == "__main__":
    all_sp500_companies = fetch_ticker()
    sp500_companies = all_sp500_companies[:100]
    prediction_results = automate_predictions(sp500_companies)
    for item in prediction_results:
        print(item)
"""

"""
def automate_predictions(companies):
    results = []
    for company in companies:
        for year in [2024, 2025]:
            for month in range(1, 13):
                prompt = f"Please predict only the stock price for {company} from sp500 during {month}/{year}."
                chat_history = [HumanMessage(content=prompt)]
                response = generate_response(
                    user_input=prompt,
                    enable_web_search=True,
                    report_structure="",
                    max_search_queries=3,
                    chat_history=chat_history
                )
                # Extract only the relevant info (example: 'price' in the response)
                predicted_price = response.get("final_answer", {}).get("price", "N/A")
                results.append([company, year, month, predicted_price])
    return results

if __name__ == "__main__":
    all_sp500_companies = fetch_ticker()
    sp500_companies = all_sp500_companies[:100]
    prediction_results = automate_predictions(sp500_companies)

    with open('stock_predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ticker", "Year", "Month", "PredictedPrice"])
        for row in prediction_results:
            writer.writerow(row)
"""


def automation(companies: list[str], rag_file_folder: str, past_months: int = 3):
    results = []
    today = datetime.date.today()
    for company in companies:
        for i in range(past_months):
            date_point = today - relativedelta(months=i+1)
            year = date_point.year
            month = date_point.month
            prompt = f"Please predict {company} stock price, financial sentiment with its confidence score for the average stock price in first seven days of{month}/{year}."
            response = generate_experiment_response(
                user_input=prompt,
                enable_web_search=True,
                max_search_queries=3,
                rag_file_folder=rag_file_folder,
                symbols=company,
            )
            price = response.get("final_answer", {}).get("price", "N/A")
            sentiment = response.get("final_answer", {}).get("sentiment", "N/A")
            confidence_score = response.get("final_answer", {}).get("confidence_score", "N/A")
            results.append([company, year, month, price, sentiment, confidence_score])
    return results


def compare_stock_predictions(predicted_csv, actual_csv):
    # Load data from both CSV files
    df_predicted = pd.read_csv(predicted_csv)
    df_actual = pd.read_csv(actual_csv)

    # Merge the two dataframes
    df_merged = pd.merge(
        df_predicted,
        df_actual,
        on=['Ticker', 'Year', 'Month'],
        suffixes=('_pred', '_actual')
    )

    # Rename columns for clarity
    df_merged.rename(columns={
        'PredictedPrice': 'pred',
        'ActualPrice': 'actual'
    }, inplace=True)

    # Calculate metrics
    df_merged['error'] = df_merged['pred'] - df_merged['actual']
    mae = df_merged['error'].abs().mean()
    mse = (df_merged['error'] ** 2).mean()
    rmse = np.sqrt(mse)
    mape = (df_merged['error'].abs() / df_merged['actual'].abs()).mean() * 100

    # Print results
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape, "%")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df_merged['actual'], label='Actual')
    plt.plot(df_merged['pred'], label='Predicted')
    plt.title("Predicted vs Actual Stock Prices")
    plt.xlabel("Data Points")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

def compare_sentiment_predictions(predicted_csv, actual_csv):
    # Load data from both CSV files
    df_predicted = pd.read_csv(predicted_csv)
    df_actual = pd.read_csv(actual_csv)

    # Merge the two dataframes
    df_merged = pd.merge(
        df_predicted,
        df_actual,
        on=['Ticker', 'Year', 'Month'],
        suffixes=('_pred', '_actual')
    )

    # Extract predicted and actual sentiment
    y_pred = df_merged['PredictedSentiment']
    y_true = df_merged['ActualSentiment']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    confusion = confusion_matrix(y_true, y_pred)

    # Visualization
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(confusion)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--rag_file_folder", type=str, required=True, help="Path to the RAG file folder")
    argparser.add_argument("-t", "--fetch_type", type=str, choices=["sp500", "sp100"], default="sp100", help="Type of companies to fetch")
    args = argparser.parse_args()
    rag_file_folder = args.rag_file_folder
    
    fetch_type = args.fetch_type
    tickers = fetch_ticker(fetch_type)

    past_months_data = automation(tickers, rag_file_folder, past_months=3)
    with open('stock_prices.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ticker", "Year", "Month", "Price", "Sentiment", "ConfidenceScore"])
        for row in past_months_data:
            writer.writerow(row)

    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("-o", "--network_ouput", type=str, required=True, help="Result generated by the network")
    # argparser.add_argument("-g", "--ground_truth", type=str, required=True, help="Ground truth csv file path")
    # args = argparser.parse_args()
    # compare_stock_predictions(args.network_output, args.ground_truth)