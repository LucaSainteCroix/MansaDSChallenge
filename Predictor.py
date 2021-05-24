import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import warnings
import math
from statsmodels.api import OLS
from datetime import datetime
import pickle5 as pickle
warnings.filterwarnings("ignore")



# Removal of extreme outliers (5 standard deviations away from the mean)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df['amount']<0]
    df['amount'] = abs(df['amount'])
     
    df["mean"] = np.mean(df['amount'])
    df["std"] = np.std(df['amount'])

    df["outlier"] = df["amount"] > df["mean"]+df["std"]*5
    df = df[~df["outlier"]]
    df["amount"] = df["amount"] * (-1)
    
    return df


# Downsampling the data by weeks

def resample_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values(by=["date", "id"], ascending=[True, False])[["date", "amount", "balance"]]

    df = df.set_index("date", drop=True)
    df = df.resample("W-MON").agg({'amount': 'sum', 'balance': 'last'}).reset_index()
    df['balance'] = df['balance'].fillna(method='ffill') # for when there's no transactions in the time window just keep the same balance


    return df


# calculate the balance after every transaction with the help of the end balance

def add_balance(df: pd.DataFrame) -> pd.DataFrame:
    df_list = []
    df['amount'] = round(df['amount'], 2)
    
    df = df.sort_values(by=["date","id"], ascending=False)
    count=0
    tmp_df = df.copy()
    for idx, row in df.iterrows():

        # we want balance before the transaction, because when trying to forecast we won't know the final balance since it's part of what we're trying to predict
        if idx != 0:
            tmp_df.balance.iloc[idx] = round(tmp_df.iloc[idx-1]["balance"] - tmp_df.iloc[idx-1]["amount"], 2) 

    df_list.append(tmp_df.drop(['update_date'], 1))
    new_df = pd.concat(df_list)
#     new_df = new_df.set_index("id")
    return new_df


# reshape data to have each week row contain data about 12 previous weeks balance and expense amount

def reshape_data(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    
    for i in range(1, 12):
        df[f"balance_{i}"] = df["balance"].shift(i)
        df[f"amount_{i}"] = df["amount"].shift(i, fill_value=0)
        
        #replace NaN values for balance, with previous balance
        if i == 1:
            df[f"balance_{i}"] = np.where(df[f"balance_{i}"].isna(), df["balance"], df[f"balance_{i}"])
        else:
            df[f"balance_{i}"] = np.where(df[f"balance_{i}"].isna(), df[f"balance_{i-1}"], df[f"balance_{i}"])
    
    df = df.rename(columns={"amount": "amount_0", "balance": "balance_0"})
    df = df.sort_values(by="date", ascending=False).iloc[:1] # only take last point of data
    
    return df


# Using our Linear Regression model to forecast one week using the 12 previous weeks data

def make_forecast(df):
    with open("week1.pickle", "rb") as f:
        model = pickle.load(f)
    
    df = df.sort_values(by="date", ascending=False)
    previous_amount_balance_columns = [column for column in df.columns if "amount" in column or "balance" in column]
    previous_amount_columns = [column for column in df.columns if "amount" in column]
    df = df[previous_amount_balance_columns]
    median_X_test = df[previous_amount_columns].median(axis=1).values[0]

    prediction = model.predict(df).values[0]

    # on rare occasions the model forecasts a positive value, in this case we take our second best guess, the median*
    if prediction > 0:
        prediction = median_X_test

    return prediction


# The function called from main.py. It gets the data from the api call, transforms it and returns the forecast. 
# No need to validate inputs here since it is done in main.py

def make_prediction(transactions, accounts, user) -> Dict[str, float]:
    df = pd.DataFrame(map(dict,transactions))
    df_user = pd.DataFrame(data={"update_date": [user.update_date], "business_NAF_code": [user.business_NAF_code]}, index = [user.id])
    df_accounts = pd.DataFrame(map(dict,accounts))
    df['date'] = pd.to_datetime(df['date'])

    df = remove_outliers(df)
    
    user_accounts = df_accounts.join(df_user, 'user_id').set_index('id')
    user_accounts_transactions = df.join(user_accounts, 'account_id').rename_axis('id')

    user_accounts_transactions = add_balance(user_accounts_transactions)

    df = resample_monthly_income(user_accounts_transactions)

    if df.shape[0]<12:
        forecast = 0.0
    else:
        df = reshape_data(df)
        forecast = round(make_forecast(df), 2)
    
    return forecast

