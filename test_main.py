import json
from datetime import datetime
from random import random
import requests


# You can use this function to test your api
# Make sure the uvicorn server is running locally on `http://127.0.0.1:8000/`
# or change the hardcoded localhost below
def test_predict():
    """
    Test the predict route with test data
    """
    test_user = {
        "id": 1,
        "update_date": str(datetime(2020, 9, 1)),
        "business_NAF_code": "7022Z",
    }
    test_accounts = [{"id": 1, "balance": 10000, "user_id": 1}]
    test_transactions = [
        {"account_id": 1, "date": str(datetime(2019, i//30+3, i%30+1)), "amount": -i*random()}
        for i in range(1, 200)
    ]
    

    test_data = {
        "user": test_user,
        "accounts": test_accounts,
        "transactions": test_transactions,
    }

    print("Calling API with test data:")
    print(test_data)

    response = requests.post(
        "http://127.0.0.1:8000/predict", data=json.dumps(test_data)
    )

    print("Response: ")
    print(response.text)

    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()
