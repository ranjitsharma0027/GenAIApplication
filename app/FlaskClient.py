"""
@author: Ranjit S.
"""

import json
import requests
import time

# Function for posting data
def post_documents():
    # Payload data for POST request
    payload = {
        "Doc1": "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
        "Doc2": "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
        "Doc3": "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
    }

    # Convert the payload to JSON
    data = json.dumps(payload)

    # POST request
    response = requests.post("http://127.0.0.1:5000/docs", data=data)

    # Check the response
    if response.status_code == 200:
        print("Documents posted successfully!")
        print("Response Data:", response.json())
    else:
        print(f"Failed to post documents. Status code: {response.status_code}")
        print("Error:", response.text)


# Function for retrieving data
def retrieve_questions(question):
    # Questions for GET request
    
    # Convert question to JSON
    data = json.dumps(question)
    print()
    print("Query Data : " , data)
    print()
    
    # GET request
    response = requests.get("http://127.0.0.1:5000/docs", data=data)

    # Check the response
    if response.status_code == 200:
        print("Questions retrieved successfully!")
        print("Response Data:", response.json())
    else:
        print(f"Failed to retrieve questions. Status code: {response.status_code}")
        print("Error:", response.text)

# Main function to execute methods
if __name__ == "__main__":
        
    print("\n Docs Processing...") 
    post_documents()
    time.sleep(10)
    
    print("\nRetrieving questions...")
    question = [{'Question': 'Tell something about the car Tiago'}]
    
    retrieve_questions(question)
    print("\n Done..")
    
