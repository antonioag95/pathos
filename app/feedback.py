import os
from dotenv import load_dotenv

# Local imports
from models import FeedbackAPI

load_dotenv()

class Feedback:
    """
    Class to handle data management in a separate collection in MongoDB
    """
    def __init__(self, db):
        self.db = db
        # Define the collection name for your new data
        FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION")
        self.feedback_collection = self.db[FEEDBACK_COLLECTION]

    def save_data(self, data):
        """
        Save data to the MongoDB collection
        """
        self.feedback_collection.insert_one(data)
        return 

    def retrieve_data(self, query):
        """
        Retrieve data from the MongoDB collection
        """
        # Implement your logic to retrieve data from the collection
        pass