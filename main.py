from datetime import datetime
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, validator
from fastapi.responses import JSONResponse
from Predictor import make_prediction

class User(BaseModel):
    update_date: datetime
    business_NAF_code: str
    id: int


class Account(BaseModel):
    user_id: int
    balance: float
    id: int


class Transaction(BaseModel):
    account_id: int
    amount: float
    date: datetime


class RequestPredict(BaseModel):
    user: User
    accounts: List[Account]
    transactions: List[Transaction]

    @validator("transactions")
    def validate_transaction_history(cls, v, *, values):
        # validate that
        # - the transaction list passed has at least 6 months history
        # - no transaction is posterior to the user's update date
        if len(v) < 1:
            raise ValueError("Must have at least one Transaction")

        update_t = values["user"].update_date

        oldest_t = v[0].date
        newest_t = v[0].date
        for t in v[1:]:
            if t.date < oldest_t:
                oldest_t = t.date
            if t.date > newest_t:
                newest_t = t.date

        assert (
            update_t - newest_t
        ).days >= 0, "Update Date Inconsistent With Transaction Dates"
        assert (update_t - oldest_t).days > 183, "Not Enough Transaction History"

        return v


class ResponsePredict(BaseModel):
    user_id: int
    predicted_amount: float


def predict(
    transactions: List[Transaction], accounts: List[Account], user: User
) -> float:
    forecast = make_prediction(transactions, accounts, user)
    return forecast

app = FastAPI()



@app.exception_handler(Exception)
async def validation_exception_handler(request, err):
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    return JSONResponse(status_code=400, content={"message": f"{base_error_message}. Detail: {err}"})

@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    accounts = predict_body.accounts
    user = predict_body.user

    # Call your prediction function/code here
    ####################################################
    predicted_amount = predict(transactions, accounts, user)

    # Return predicted amount along with account id
    return {"user_id": user.id, "predicted_amount": predicted_amount}
