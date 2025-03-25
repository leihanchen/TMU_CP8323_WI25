import csv
import streamlit as st
from app import generate_response, fetch_ticker
from langchain_core.messages import HumanMessage

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