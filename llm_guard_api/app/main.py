import os
import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llm_guard.vault import Vault
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize the Vault
vault = Vault()

# Configure the Anonymize Scanner
scanner = Anonymize(
    vault,
    preamble="Insert before prompt",
    allowed_names=["John Doe"],
    hidden_names=["Test LLC"],
    recognizer_conf=BERT_LARGE_NER_CONF,
    language="en"
)

# Logger
LOGGER = structlog.getLogger(__name__)

# Request model for prompt sanitization
class SanitizePromptRequest(BaseModel):
    prompt: str

# Root endpoint
@app.get("/", tags=["Main"])
def read_root():
    return {"name": "LLM Guard API"}

# Health check endpoint
@app.get("/healthz", tags=["Health"])
def read_healthcheck():
    return JSONResponse({"status": "alive"})

# Sanitize Prompt endpoint
@app.post('/sanitize-prompt', tags=["Sanitization"])
async def sanitize_prompt(request: SanitizePromptRequest):
    prompt = request.prompt

    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Scan and sanitize the prompt
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

    return {
        'sanitized_prompt': sanitized_prompt,
        'is_valid': is_valid,
        'risk_score': risk_score
    }
