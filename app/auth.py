# Standard library imports
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

# Third-party imports
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status

# Local imports
from models import User

logging.getLogger("passlib").setLevel(logging.ERROR)

# Load environment variables from a .env file
load_dotenv()

class Auth():
    """
    Class to handle FastAPI server authentication using JWT authentication
    """
    def __init__(self):
        # Set secret key, algorithm, and expiry time
        self.SECRET_KEY = "1d07ff195af3f7f222ac883c0ad16fcdcf9665801c7177159f37cf33ee8cd5f7" # openssl rand -hex 32
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        # Create a hashing context to securely hash and check passwords
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Connect to MongoDB
        MONGO_URI = os.getenv("MONGO_URI")
        DB_NAME = os.getenv("DB_NAME")
        USER_COLLECTION = os.getenv("USER_COLLECTION")
        DB_USER = urllib.parse.quote_plus(os.getenv("DB_USER"))
        DB_PWD = urllib.parse.quote_plus(os.getenv("DB_PWD"))

        try:
            self.client = MongoClient(f"mongodb://{DB_USER}:{DB_PWD}@{MONGO_URI}/?authMechanism=DEFAULT&tls=true")
            self.db = self.client[DB_NAME]
            self.users_collection = self.db[USER_COLLECTION]
            logging.info("Successfully connected to MongoDB.")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")

    # Define a function to hash a password
    def hash_password(self, password: str):
        """
        Returns a hashed password using the pwd_context
        """
        return self.pwd_context.hash(password)

    # Define a function to check a password hash
    def verify_password(self, password: str, hashed_password: str):
        """
        Checks a password hash against the password
        """
        return self.pwd_context.verify(password, hashed_password)

    # Define a function to create a new JWT access token
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """
        Creates an access token with expiry, allows for custom expiry time, otherwise default expiry time
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    # Define a function to decode and verify a JWT access token
    def decode_access_token(self, token: str):
        """
        Decodes the access token using the secret key, returns a HTTPException if the token is invalid
        """
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    def authenticate_user(self, user: str, password: str):
        """
        Checks the user exists then authenticates using password.
        Returns User if authenticated successfully, otherwise returns None.
        """
        user_dict = self.users_collection.find_one({"$or":[ {"username":user}, {"email":user}]})
        if user_dict is None:
            return None
        if not self.verify_password(password, user_dict["hashed_password"]):
            return None
        user = User(**user_dict)
        return user
    
    def get_user(self, user: str):
        """
        Retrieves a user from the database by user.
        Returns the User object if found, otherwise returns None.
        """
        query = self.users_collection.find_one({"$or":[ {"username":user}, {"email":user}]})
        if query and (query.get("username") == user or query.get("email") == user):
            user_dict = query
            return User(**user_dict)
        return None