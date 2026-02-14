import logging
import os
from datetime import datetime

# Create logs directory
os.makedirs("logs", exist_ok=True)

LOG_FILE = f"logs/pipeline_{datetime.now().date()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AQI_PIPELINE")
