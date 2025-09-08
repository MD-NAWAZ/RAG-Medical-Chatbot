# from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token : str = HF_TOKEN):
    try:
        logger.info("Loading LLM from Huggingface")

        # llm = HuggingFaceEndpoint(
        #     repo_id = huggingface_repo_id,
        #     huggingfacehub_api_token = hf_token,
        #     temperature = 0.3,
        #     max_new_tokens=256,
        #     return_full_text = False
            
        # )
        llm = ChatGroq(api_key=GROQ_API_KEY,model="llama-3.1-8b-instant",temperature=0)

        logger.info("LLM loaded Successfully")

        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load the LLM",e)
        logger.error(str(error_message))

        
