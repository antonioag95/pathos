# FastAPI and related imports
import logging
import uvicorn
import pathlib
from fastapi import FastAPI, HTTPException, Depends, Request, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse, RedirectResponse
from typing import Annotated
from models import User, InputAPI, BaseResponse, FeedbackAPI
from auth import Auth
from feedback import Feedback

# Time and datetime-related imports
import time
from datetime import datetime

# Numerical and deep learning imports
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

###########################################################################
logging.info('Pathos is starting up')

class OAuth2PasswordCookie(OAuth2PasswordBearer):
    """OAuth2 password flow with token in a httpOnly cookie.
    """

    def __init__(self, *args, token_name: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_name = token_name or "my-jwt-token"

    @property
    def token_name(self) -> str:
        """Get the name of the token's cookie.
        """
        return self._token_name

    async def __call__(self, request: Request) -> str:
        """Extract and return a JWT from the request cookies.

        Raises:
            HTTPException: 403 error if no token cookie is present.
        """
        token = request.cookies.get(self._token_name)
        if not token:
            # If token is not found raise HTTPException
            raise HTTPException(status_code=401, detail="Invalid access token")
        return token
        
# FastAPI instance
app = FastAPI(docs_url=None, redoc_url=None)
# Auth instance
auth = Auth()
# OAuth2 scheme
oauth2_scheme = OAuth2PasswordCookie(token_name="access_token", tokenUrl="token")
# Feedback instance
feedback = Feedback(auth.db)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and tokenizer paths
model_path = "./models/BERT_ft_epoch3_TestScore_0.9167767129186862.bin"
tokenizer_path = "google-bert/bert-base-multilingual-uncased"

# Emotion labels mapping
emotion_labels = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Instantiate the model configuration
config = BertConfig.from_pretrained(tokenizer_path, num_labels=6)

# Instantiate the model class with the configuration
model = BertForSequenceClassification(config)

# Load the state_dict into the model
# map_location=torch.device(device) dynamically selects either CPU or GPU
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

###########################################################################
# Custom exception handler for 400 errors
# This function is responsible for handling HTTPException instances with a status code of 400.
@app.exception_handler(400)
async def bad_request_exception_handler(request: Request, exc: HTTPException):
    logging.warning(f"{get_client_ip(request)} - {request.url.path} - {exc.detail}")
    # Return a JSON response with a 400 status code and the detail message from the HTTPException.
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": exc.detail})

# Custom exception handler for 401 errors
# This function is responsible for handling HTTPException instances with a status code of 401.
@app.exception_handler(401)
async def not_authorized_exception_handler(request: Request, exc: HTTPException):
    logging.warning(f"{get_client_ip(request)} - {request.url.path} - {exc.detail}")
    # Return a JSON response with a 401 status code and the detail message from the HTTPException.
    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"message": exc.detail})

# Custom exception handler for 404 errors
# This function is responsible for handling HTTPException instances with a status code of 404.
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    logging.error(f"{get_client_ip(request)} - {request.url.path} - {exc.detail}")
    # Return a JSON response with a 404 status code and a custom message indicating the resource was not found.
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "404 not found"})

def get_model_name(model_name: str):
    return pathlib.Path(model_name).stem

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
#async def get_current_user(access_token: str = Cookie(...)):
    """
    Decodes an access token, and authenticates user.
    Returns 401 if access token is invalid.
    """
    payload = auth.decode_access_token(token)
    user = payload.get("sub")
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = auth.get_user(user=user)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Checks if user is active, allows for turning on and off of individual users.
    """
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    Logs in to return an access token when supplied with valid username and pass. 
    Returns 400 if login details incorrect.
    """
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect user or password")
    access_token = auth.create_access_token(
        data={"sub": user.email}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to extract client's IP address
def get_client_ip(request: Request):
    """
    Retrieve the IP address of the client making the HTTP request.

    Parameters:
        request (Request): The FastAPI Request object representing the incoming HTTP request.

    Returns:
        str: The IP address of the client making the request.
    """
    return request.client.host

def get_client_user_agent(request: Request):
    """
    Function to retrieve the User-Agent header from an HTTP request.

    Args:
    - request: An instance of the Request class from FastAPI.

    Returns:
    - user_agent (str or None): The User-Agent header value if present in the request headers, otherwise None.
    """
    # Retrieve the User-Agent header from the request
    user_agent = request.headers.get("user-agent")

    # Now you can use the user_agent variable as needed
    if user_agent:
        return user_agent
    else:
        return None
    
# Dependency to generate base response
async def get_base_response(request: Request):
    """
    Generate a base response containing common keys for an HTTP request.

    Parameters:
        request (Request): The FastAPI Request object representing the incoming HTTP request.

    Yields:
        BaseResponse: An instance of BaseResponse with common keys populated.
    """
    # Record the start time
    start_time = time.time()

    # Get the current time in Epoch time (Unix time)
    current_epoch_time = int(datetime.now().timestamp())

    # Create an instance of BaseResponse to include common keys
    base_response = BaseResponse(timestamp=current_epoch_time)

    # Get the client's IP address and add it to the base response
    client_ip = get_client_ip(request)
    base_response.ip = client_ip

    # Get the client's User-Agent and add it to the base response
    client_user_agent = get_client_user_agent(request)
    base_response.user_agent = client_user_agent

    # Add the start time to the request state
    request.state.start_time = start_time

    # Yield the base response
    yield base_response

# Function to preprocess text and predict emotion
def predict_emotion(text):
    """
    Function to predict emotions from input text.

    Args:
    text (str): Input text to analyze.

    Returns:
    list: List of dictionaries containing probabilities of different emotions for the input text.
    """
    # Tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Convert probabilities to list of dictionaries
    probs_list = [{emotion_labels[i]: prob.item() for i, prob in enumerate(row)} for row in probabilities]

    return probs_list[0]

def infer_sentiment(emotion):
    """
    Function to infer sentiment based on emotion.

    Args:
    emotion (str): The emotion to analyze.

    Returns:
    str: Sentiment inferred from the emotion. Can be "positive" or "negative".
    """
    positive_emotions = ["joy", "love", "surprise"]  # Define a list of positive emotions
    if emotion in positive_emotions:  # Check if the input emotion is in the list of positive emotions
        return "positive"  # Return "positive" if the emotion is positive
    return "negative"  # Return "negative" otherwise

###########################################################################

# Mount the static folder under /img
app.mount("/img", StaticFiles(directory="./static/img"), name="img")

# Login endpoint    
@app.get("/login")
async def read_login(request: Request, base_response: BaseResponse = Depends(get_base_response)):
    try:
        # Attempt to retrieve the access token from the request cookies
        token = request.cookies.get("access_token")

        # If no access token is present, redirect to the login page
        # Generate the login page HTML
        login_html = FileResponse("./static/login.html")

        # Add cache-control headers to prevent caching
        login_html.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        login_html.headers["Pragma"] = "no-cache"
        login_html.headers["Expires"] = "0"
    
        if token is None:
            # If no access token is present, return the login page HTML
            logging.warning(f"{get_client_ip(request)} - No access token. Redirecting to login page")
            return login_html
        
        # Attempt to retrieve the current user using the access token
        user = await get_current_user(token)
        if user:
            # If a valid user is present, redirect to the index page
            logging.info(f"{get_client_ip(request)} - Access token is valid. Going to next page")
            return RedirectResponse(url='/', status_code=status.HTTP_303_SEE_OTHER)
        else:
            # If no valid user is present, redirect to the login page
            logging.warning(f"{get_client_ip(request)} - No valid access token. Redirecting to login page")
            return login_html
    except HTTPException as e:
        # If there's an HTTPException (e.g., invalid access token), redirect to the login page
        return login_html

# Root endpoint    
@app.get("/")
async def read_root(request: Request, base_response: BaseResponse = Depends(get_base_response)):
    try:
        # Attempt to retrieve the access token from the request cookies
        token = request.cookies.get("access_token")
        if token is None:
            # If no access token is present, redirect to the login page
            logging.warning(f"{get_client_ip(request)} - No access token. Redirecting to login page")
            return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)
        
        # Attempt to retrieve the current user using the access token
        user = await get_current_user(token)
        if user:
            # If a valid user is present, return the index page
            logging.info(f"{get_client_ip(request)} - Access token is present. Going to home")
            return FileResponse("./static/index.html")
        else:
            # If no valid user is present, redirect to the login page
            logging.warning(f"{get_client_ip(request)} - No valid access token. Redirecting to login page")
            return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException as e:
        # If there's an HTTPException (e.g., invalid access token), redirect to the login page
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

@app.post("/token")
async def auth_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], request: Request, base_response: BaseResponse = Depends(get_base_response)):
    """
    Generates an authorization token using the login function and form data.
    """
    # Create a dictionary with token data
    token = await login(form_data=form_data)

    # Calculate the response time
    end_time = time.time()
    response_time = int((end_time - request.state.start_time) * 1000)  # Convert to milliseconds
    base_response.response_ms = response_time  # Assign the response time to base_response object

    # Update the response dictionary with additional information from the base response
    # We exclude fields with default values (None) to prevent them from appearing in the response
    token["info"] = base_response.model_dump(exclude_unset=True)

    # Redirect to the homepage
    response = RedirectResponse(url='/', status_code=status.HTTP_302_FOUND)

    # We can now save the HTTP cookie containing the access token
    response.set_cookie(key="access_token", value=token["access_token"], httponly=True)

    # Return the response dictionary
    return response

@app.get("/api/whoami/")
async def read_own_items(current_user: Annotated[User, Depends(get_current_active_user)], request: Request, base_response: BaseResponse = Depends(get_base_response)):
    
    # Prepare dictionary to return
    response = {"logged_user": current_user.username, "logged_email": current_user.email}

    # Update the response dictionary with the values from the base response under 'info'
    # We exclude fields with default values (None) to prevent them from appearing in the response
    response["info"] = base_response.model_dump(exclude_unset=True)
    
    return response

# Endpoint to predict emotion in English/Italian/Portuguese text
@app.post("/api/emotion/")
async def create_item(input: InputAPI, request: Request, current_user: Annotated[User, Depends(get_current_active_user)], base_response: BaseResponse = Depends(get_base_response)):
    # Get input text and whether to return probabilities
    text = input.text.strip()
    return_probabilities = input.probs

    # Predict emotion probabilities
    emotion_probabilities = predict_emotion(text)

    # Convert probabilities to predicted label
    predicted_label = max(emotion_probabilities, key=emotion_probabilities.get)

    # Infer the sentiment from the emotion
    infered_sentiment = infer_sentiment(predicted_label)

    # Prepare dictionary to return
    response = {"text": text, "emotion": predicted_label, "sentiment": infered_sentiment}

    # If requested, include probabilities in the response
    if return_probabilities:
        response["probabilities"] = emotion_probabilities

    # Calculate the response time
    end_time = time.time()
    response_time = int((end_time - request.state.start_time) * 1000)  # Convert to milliseconds
    base_response.response_ms = response_time

    # Add the logged user info
    base_response.username = current_user.username
    base_response.email = current_user.email

    # Assigning model name to base_response object
    base_response.model = get_model_name(model_path)

    # Assigning inference device information to base_response object
    base_response.device = str(device)

    # Update the response dictionary with the values from the base response under 'info'
    response["info"] = base_response.model_dump(exclude_unset=True)

    # Log the results
    logging.info(f"{get_client_ip(request)} - {request.url.path} - {str(response)}")

    # Return the dictionary containing text, predicted emotion, response time, Epoch time, and client's IP address
    return response

@app.post("/api/feedback/")
async def provide_prediction_feedback(input: FeedbackAPI, current_user: Annotated[User, Depends(get_current_active_user)], request: Request, base_response: BaseResponse = Depends(get_base_response)):
    # Extract the feedback data from the request
    feedback_sentence = input.text.strip()
    feedback_sentiment = input.sentiment.strip()
    feedback_emotion = input.emotion.strip()

    # Create a response dictionary with the feedback data and user information
    response = {
        "text": feedback_sentence,
        "sentiment": feedback_sentiment,
        "emotion": feedback_emotion,
        "feedback_user": current_user.username,  # Add the username of the current user
        "feedback_email": current_user.email     # Add the email of the current user
    }

    # Log the feedback
    logging.info(f"{get_client_ip(request)} - {request.url.path} - {str(response)}")
    
    # Save the feedback data to the database
    feedback.save_data(response)

    # Remove the "_id" field from the response dictionary (pyMongo auto adds it)
    response = {key: value for key, value in response.items() if key != "_id"}
        
    # Add the logged user info to the base response
    base_response.username = current_user.username
    base_response.email = current_user.email

    # Update the response dictionary with the values from the base response under 'info'
    # Exclude fields with default values (None) to prevent them from appearing in the response
    response["info"] = base_response.model_dump(exclude_unset=True)
    
    # Return the response dictionary
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_config="log.ini")