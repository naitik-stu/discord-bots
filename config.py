import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Bot settings
    BOT_PREFIX = "!"
    TICKET_CHANNEL_ID = int(os.getenv('TICKET_CHANNEL_ID', 0))
    LOG_CHANNEL_ID = int(os.getenv('LOG_CHANNEL_ID', 0))
    
    # AI settings
    MODEL_NAME = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.7
    MAX_RESPONSE_LENGTH = 500
    
    # Training data
    DATA_FILE = "training_data.txt"
    VECTOR_DB_FILE = "vectors.faiss"
