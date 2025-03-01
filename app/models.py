from typing import List, Optional
from datetime import date
from pydantic import BaseModel, EmailStr


class User(BaseModel):
    """
    User class containing information on users, will be used to store users in a DB
    """
    name: str = ""
    surname: str = ""
    birthdate: Optional[date] = None
    username: str
    email: str
    hashed_password: str
    disabled: bool = False

class UserSignupAPI(BaseModel):
    """
    Pydantic model for user signup requests.
    """
    name: str
    surname: str
    birthdate: date
    username: str
    email: EmailStr
    password: str

# Data model for input to the API
class InputAPI(BaseModel):
    # Define text field for input text
    text: str
    # Define probs field for indicating whether to return probabilities (default to False if not provided)
    probs: bool | None = False

# Base response model containing common keys
class BaseResponse(BaseModel):
    username: str | None = None
    email: str | None = None
    timestamp: int | None = None
    ip: str | None = None
    user_agent: str | None = None
    response_ms: int | None = None
    model: str | None = None
    device: str | None = None

# Data model for Feedback API   
class FeedbackAPI(BaseModel):
    text: str
    sentiment: str
    emotion: str
    