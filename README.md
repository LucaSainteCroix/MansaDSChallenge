# My Participation to the Mansa data coding challenge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is my contribution for the Mansa Data Science Coding Challenge.

I separated the notebooks into 3: Exploratory Data Analysis, Feature Engineering and finally Model Selection.

All the functions used by the FastAPI to forecast are in the Predictor.py file, which is imported in main.py.
To run it, configure a Python 3 virtual environment and run `pip install -r requirements.txt`, then `uvicorn main:app` to launch the local server, and to test, run `python test_main.py`.

I chose to predict weekly expenses, because there is some monthly seasonality, and, forecasting from 12 weekly observations is better than from 6 monthly observations.

It was my first time working on a Time Series project. Even though I tried a lot of different approaches, I ended up using a linear regressor because it had the best results and is fast to implement. But I also could have implemented the Prophet solution I tested in the 3rd notebook, which had decent results.





## Goal:

To test your ability to deal with time-series data, build a model, and serve a model using an API framework.


## Details:

In the `data` folder, you will find three `csv` files containing real anonymized data from transactions, accounts and users. The `transactions.csv` contains a set of transactions with an amount (in eur) and the date of the day they were added. The `accounts.csv` contains a list of balances for the accounts that the transactions pertain to. The `users.csv` contains a list of the account owners with the date of the last update of their financial data, as well as the NAF code of the business of the user. Note that the balance provided for each account is the balance on the update date of the user the account belongs to. Additionnaly, a table of correspondance between the NAF code and the official name of the activity is provided in the `business_NAF.csv` file (more info about the NAF code can be found at https://www.insee.fr/fr/information/2406147).

Provided with this repo is also a `main.py` file with a minimal [FastAPI](https://fastapi.tiangolo.com/) demo. Once you have installed the `requirements.txt` in your python environment you will be able to run the main file by simply calling `uvicorn main:app` inside your directory. This should start the local server and you should be able to see the automatically generated API docs at `http://127.0.0.1:8000/docs`. 

## The Task

Your task is to use this data to make a model capable of making predictions. There are two options for models that you can build - pick and implement the one you feel is more fun. 

Your model should then be served through the small [FastAPI](https://fastapi.tiangolo.com/) file provided. Feel free to extend it or split the file as needed.

### Before you pick a model/approach:
We suggest you take a look and explore the data. Since it is real data, it is noisy and sparse. Some columns such as dates might need to be parsed as such. 

Build a function to check which accounts have more than 180 days of history - you can discard the others for your models and analysis. *You can assume that any accounts/transactions passed to your service will have at least 6 months of history.*

You have several options for the model to build:

### Option A: Predict the next month's income given the past six months

Set up a prediction function that takes a list of accounts, a list of transactions and a user and ouputs a prediction for the aggregated next monthly income of that user. 

Income is defined as the sum of all transactions with the amount `> 0` over a certain time-period. So to get the monthly income, you can sum the transactions over monthly periods. 

Tip: It might make more sense to define a month as a 30 day period rather than the month itself since the snapshots can be taken at any point during the month and not necessarily at the end. Alternatively you could decide to only keep "full" months of transactions in the dataset.  

### Option B: Predict the next week's outgoing given the past 3 months (12 weeks)

Set up a prediction function that takes a list of accounts, a list of transactions and a user and ouputs a prediction for the aggregated next weekly outgoing for that user. 

Outgoing is defined as the sum of all transactions with the amount `< 0` over a certain time-period. So to get the weekly outgoing, you can sum all the negative transactions within a 7 day (or week) period.


### For both options:

The preprocessing/algorithms/loss functions are yours to decide, as well as the separation between train/validation/test set. You do not necessarily have to use all the data provided either if you feel like some of it is irrelevant or not useful, as long as you can justify the choices that you make.

Specific design choices must be justified, whether that be qualitatively through plots or quantitatively through metrics. We want you to explain to us why you make the choices that you make, and to show us how good your solution is.

Tip: By combining the transactions and accounts data you should be able to reverse the balance of the account back through time (back to the oldest transaction date for the account). This information might be useful depending on what you choose to predict.


### Serving your solution through FastAPI

Once you have a model implemented which can predict the given output, we would like you to try to serve it in the API file provided (`main.py`).
 
This requires you to move your preprocessing to a function which you can call any input on. **You do not have to worry about validating the inputs - FastAPI will do this for you!**

You can then move your predict functionality to the `predict` function and return the predicted amount.
 

If you use `pandas`, you can convert the `transactions`  of type `List[Transaction]` (or similarly the `accounts` of type `List[Account]`)  passed to the API to a `pd.DataFrame` by calling:

```python
import pandas as pd 

df = pd.DataFrame(map(dict,transactions))
``` 

This is because the objects passed to the API are using pydantic's `BaseModel` class which allows easy conversion from object to dictionary through the default `.dict()` implementation.

You can test your API using the `test_main.py` file, just make sure you are running the server by calling `uvicorn main:app` in another terminal window.

If you wish to learn more about how to use FastAPI:

- [Official FastAPI Docs](https://fastapi.tiangolo.com/)
- [Official pydantic Docs](https://pydantic-docs.helpmanual.io/)
- [Medium Post: How to Deploy a Machine Learning Model](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-dc51200fe8cf)

## What we expect:

- Use of Python (3.X)
- Clearly documented code or explanations with each function. **You need to be able to justify your design choices** - from data processing to algorithm decisions.
- Use of the FastAPI format for routes and for serving your prediction model and use of pydantic/typing for input/output validation (this is free 😉)

What we really care about is getting to know your thought process, decision making and data science reflexes! 
We are not going to penalize you for accuracy, but we would encourage you to discuss the strengths and weaknesses of your approach, and possible ways to improve your predictions.

You can use whatever other external software libraries you think are appropriate. Pandas/numpy/scikit-learn are encouraged!

Your solution must be able to run and respond to requests (it can take as long to calculate as you want). You can imagine it as a micro-service that could be run independently on a server. Additional notebooks, analysis or plots to accompany your model will be very welcomed!


#### To Submit

Send an email to: remy.tincoATgetmansa.com and benjamin.cambierATgetmansa.com. 

You can send us a github link to your solution (preferred) or a zipped file with your code and explanations.
We look forward to your solution 🙂

### Why kuebiko ?

Kuebiko is the japanese god of knowledge. Early on in our creation, we took to naming our "katas" or coding challenges after japanese mythology. 

We hope to challenge you as you will be challenged in your daily work at Mansa and so designed this test with this in mind. It is important to us to gather feedback on this process as well, and so please don't hesitate to let us know what you think!
