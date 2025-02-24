from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
from llm_guard.vault import Vault 
from llm_guard.input_scanners import BanTopics, Anonymize, BanCode, BanCompetitors, BanSubstrings, Code, Gibberish, InvisibleText, Language, PromptInjection, Regex, Secrets, Sentiment, TokenLimit, Toxicity
from llm_guard.input_scanners.toxicity import MatchType
from llm_guard.input_scanners.regex import MatchType
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.input_scanners.language import MatchType
from llm_guard.input_scanners.gibberish import MatchType
from llm_guard.input_scanners.ban_substrings import MatchType
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from llm_guard.output_scanners import BanCompetitors, BanTopics, Deanonymize, Code, BanCompetitors, LanguageSame, Regex, Sentiment, Toxicity, Bias, MaliciousURLs, NoRefusal, NoRefusalLight, FactualConsistency, Relevance, Sensitive
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

from enum import Enum
from dotenv import load_dotenv
load_dotenv()
# Initialize FastAPI app
vault = Vault()
app = FastAPI()
# Define the request model for the prompt
# Initial competitor list
competitor_list = ["A1"]
# Sample competitors' names to be banned
competitors_names = [
    "Acorns",
    "Citigroup",
    "Citi",
    "Fidelity Investments",
    "Fidelity",
    "JP Morgan Chase and company",
    "JP Morgan",
    "JP Morgan Chase",
    "JPMorgan Chase",
    "Chase",
    "M1 Finance",
    "Stash Financial Incorporated",
    "Stash",
    "Tastytrade Incorporated",
    "Tastytrade",
    "ZacksTrade",
    "Zacks Trade",
]
# Initialize the BanSubstrings scanner with the competitors' names
bansubstr_scanner = BanSubstrings(
    substrings=competitors_names,
    match_type=MatchType.STR,  # Match substrings at the string level
    case_sensitive=False,
    redact=False,
    contains_all=False,  # Check for any of the substrings
)

# Initialize the BanCompetitors scanner with the initial competitor list
scanner = BanCompetitors(competitors=competitor_list, redact=False, threshold=0.5)

# Define the input model for the prompt data
class PromptRequest(BaseModel):
    prompt: str

# Define the input model for updating the competitor list
class CompetitorListRequest(BaseModel):
    competitors: List[str]
class BanTopicsRequest(BaseModel):
    prompt: str
    topics: list
    threshold: float
class AnonymizeRequest(BaseModel):
    prompt: str
    allowed_names: list
    hidden_names: list
    preamble: str
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_guard.input_scanners import BanCode

# Initialize FastAPI app
app = FastAPI()

# Define the input model to validate the prompt data
class PromptRequest(BaseModel):
    prompt: str

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_guard.output_scanners import BanTopics

app = FastAPI()

# Define request models
class BanTopicsRequest(BaseModel):
    """
    Request model for input scanning.
    Attributes:
        prompt (str): The input prompt to scan for banned topics.
        topics (list[str]): List of banned topics to check against.
        threshold (float): The risk threshold for classification.
    """
    prompt: str
    topics: list[str] = ["violence"]  # Default sensitive topic
    threshold: float = 0.5  # Default risk threshold

class BanTopicsOutputRequest(BaseModel):
    """
    Request model for output scanning.
    Attributes:
        prompt (str): The original input prompt.
        model_output (str): The generated model output to scan.
        topics (list[str]): List of banned topics to check against.
        threshold (float): The risk threshold for classification.
    """
    prompt: str
    model_output: str
    topics: list[str] = ["violence"]
    threshold: float = 0.5

# Input Scanner Endpoint
@app.post('/ban-topics', tags=["Ban Topics"])
async def ban_topics(request: BanTopicsRequest):
    """
    Scans the input prompt for banned topics using a zero-shot classifier.
    Returns a success message and risk score if valid.
    """
    prompt = request.prompt
    topics = request.topics
    threshold = request.threshold
    
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    scanner = BanTopics(topics=topics, threshold=threshold)
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Prompt contains banned topics")
    
    return {"message": "Prompt sanitized successfully", "risk_score": risk_score}

# Output Scanner Endpoint
@app.post('/ban-topics-output', tags=["Ban Topics"])
async def ban_topics_output(request: BanTopicsOutputRequest):
    """
    Scans the generated model output for banned topics using a zero-shot classifier.
    Returns a success message and risk score if valid.
    """
    prompt = request.prompt
    model_output = request.model_output
    topics = request.topics
    threshold = request.threshold
    
    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    scanner = BanTopics(topics=topics, threshold=threshold)
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Output contains banned topics")

    return {
        "message": "Output sanitized successfully",
        "risk_score": risk_score
    }

class AnonymizationRequest(BaseModel):
    """
    Represents a request body for anonymizing or deanonymizing the prompt and model output.
    
    Attributes:
        prompt (str): The sanitized prompt (which may contain placeholders).
        model_output (str): The model's output (which may contain placeholders).
    """
    prompt: str
    model_output: str


@app.post('/anonymize', tags=["Anonymization"])
async def anonymize(request: AnonymizeRequest):
    """
    Anonymizes a given prompt by replacing sensitive names with placeholders.
    The names to be anonymized and allowed names are specified in the request body.
    Args:
        request (AnonymizeRequest): The request object containing the prompt, names to hide, names to allow, and preamble.
    Returns:
        dict: A response with the sanitized prompt, success message, and risk score.
    Raises:
        HTTPException: If the prompt contains unpermitted names or is invalid.
    """
    prompt = request.prompt
    allowed_names = request.allowed_names
    hidden_names = request.hidden_names
    preamble = request.preamble
    
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    # Initialize the Anonymize scanner
    scanner = Anonymize(
        vault,
        preamble=preamble,
        allowed_names=allowed_names,
        hidden_names=hidden_names,
        recognizer_conf=BERT_LARGE_NER_CONF,
        language="en"
    )
    
    # Scan and anonymize the prompt
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Prompt contains unpermitted names")
    
    # If anonymized successfully, return the sanitized prompt and risk score
    return {"sanitized_prompt": sanitized_prompt, "message": "Prompt anonymized successfully", "risk_score": risk_score}

@app.post('/deanonymize', tags=["Deanonymization"])
async def deanonymize(request: AnonymizationRequest):
    """
    Deanonymizes the model output by replacing placeholders with original values.
    
    Args:
        request (AnonymizationRequest): The request object containing sanitized prompt and model output.
    
    Returns:
        dict: A response with the sanitized model output and risk score.
        
    Raises:
        HTTPException: If the model output contains unpermitted placeholders.
    """
    sanitized_prompt = request.prompt
    model_output = request.model_output

    if not sanitized_prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    # Initialize the Deanonymize scanner with Vault
    scanner = Deanonymize(vault)

    # Scan and deanonymize the model output
    sanitized_model_output, is_valid, risk_score = scanner.scan(sanitized_prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains unpermitted placeholders")

    # Return the deanonymized model output
    return {
        "sanitized_model_output": sanitized_model_output,
        "message": "Model output deanonymized successfully",
        "risk_score": risk_score
    }


class CodeScanRequest(BaseModel):
    """
    Represents a request body for scanning both the sanitized prompt and
    the model output to identify code snippets in banned programming languages.
    
    Attributes:
        prompt (str): The sanitized prompt.
        model_output (str): The model's output.
        languages (List[str]): A list of languages whose code snippets should be banned.
        is_blocked (bool): Whether to block or allow the listed languages.
    """
    prompt: str
    model_output: str
    languages: List[str]  # Languages to ban
    is_blocked: bool  # Whether to block the specified languages



# Create the endpoint to anonymize text input
@app.post("/bancode", tags=["Ban Code"])
async def scan_prompt(request: PromptRequest):
    prompt = request.prompt
    
    try:
        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        # Return the response as a JSON object
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post('/ban-code-output', tags=["Ban Code"])
async def ban_code(request: CodeScanRequest):
    """
    Scans both input prompt and model output for code snippets in specific languages
    and blocks them if they are present.

    Args:
        request (CodeScanRequest): The request object containing prompt, model output,
                                    and a list of banned languages.

    Returns:
        dict: A response with success message, risk score, and sanitized output.

    Raises:
        HTTPException: If the model output contains banned code snippets.
    """
    prompt = request.prompt
    model_output = request.model_output
    languages = request.languages
    is_blocked = request.is_blocked

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    # Initialize the Code scanner with the specified languages and blocking configuration
    scanner = Code(languages=languages, is_blocked=is_blocked)

    # Scan both the prompt and model output for code snippets in the banned languages
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains banned code snippets")

    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }



class CompetitorScanRequest(BaseModel):
    """
    Represents a request body for scanning both the prompt and the model output 
    to identify mentions of competitors and handle them accordingly.
    
    Attributes:
        prompt (str): The input prompt for the model.
        model_output (str): The output generated by the model.
        competitors (List[str]): A list of competitors' names to be flagged or redacted.
        redact (bool): Whether to redact competitors' names or just flag them.
        threshold (float): A risk threshold for how confident the model should be before flagging competitors.
    """
    prompt: str
    model_output: str
    competitors: List[str]  # List of competitor names or variations
    redact: bool  # Whether to redact the competitor's name
    threshold: float = 0.5  # Threshold for entity detection confidence


@app.post("/sanitize_competitors", tags=["Ban Competitors"])
async def scan_competitors(request: BaseModel):
    prompt = request.dict().get("prompt")
    output = request.dict().get("output")

    if not prompt or not output:
        raise HTTPException(status_code=400, detail="Prompt and output are required")

    try:
        # Initialize the BanCompetitors scanner
        scanner = BanCompetitors(competitors=competitor_list, redact=False, threshold=0.5)

        # Perform the scan
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, output)

        return {
            "sanitized_output": sanitized_output,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to scan for competitors
@app.post("/scan_competitors", tags=["Competitors"])
async def scan_prompt(request: PromptRequest):
    prompt = request.prompt
    
    try:
        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        # Return the response as a JSON object
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/ban-competitors-output', tags=["Ban Competitors"])
async def ban_competitors(request: CompetitorScanRequest):
    """
    Scans both input prompt and model output for mentions of competitors.
    The function will flag or redact any references to competitors based on user preference.
    
    Args:
        request (CompetitorScanRequest): The request body containing prompt, model output, 
                                          competitors list, redact flag, and threshold.
    
    Returns:
        dict: A response with sanitized model output, risk score, and success message.

    Raises:
        HTTPException: If the model output contains competitors' names.
    """
    prompt = request.prompt
    model_output = request.model_output
    competitors = request.competitors
    redact = request.redact
    threshold = request.threshold

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    # Initialize the BanCompetitors scanner with provided list of competitors and redact flag
    scanner = BanCompetitors(competitors=competitors, redact=redact, threshold=threshold)

    # Scan both the prompt and model output for competitor mentions
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains competitors' names")

    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }


class MatchType(str, Enum):
    STRING = "string"  # Check the entire output for banned substrings
    WORD = "word"  # Check for whole words matching banned substrings

class BanSubstringsScanRequest(BaseModel):
    """
    Represents a request body for scanning both the prompt and model output 
    to identify banned substrings and handle them accordingly.
    
    Attributes:
        prompt (str): The input prompt for the model.
        model_output (str): The output generated by the model.
        substrings (List[str]): A list of banned substrings to be flagged or redacted.
        match_type (MatchType): The granularity of the matching - either 'string' or 'word'.
        case_sensitive (bool): Whether the matching should be case-sensitive.
        redact (bool): Whether to redact the banned substrings by replacing them with [REDACT].
        contains_all (bool): If True, all banned substrings must be present to trigger a flag.
    """
    prompt: str
    model_output: str
    substrings: List[str]  # List of banned substrings
    match_type: MatchType  # Match type: 'string' or 'word'
    case_sensitive: bool = False  # Whether the matching should be case-sensitive
    redact: bool = False  # Whether to replace substrings with [REDACT]
    contains_all: bool = False  # Whether all substrings must appear


# Route to scan the prompt for banned substrings
@app.post("/scan_banned_substrings", tags=["Ban Substrings"])
async def scan_prompt(request: PromptRequest):
    prompt = request.prompt
    
    try:
        # Perform the scan for banned substrings
        sanitized_prompt, is_valid, risk_score = bansubstr_scanner.scan(prompt)
        
        # Return the response as a JSON object
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import HTTPException, APIRouter
from llm_guard.output_scanners import BanSubstrings
from llm_guard.input_scanners.ban_substrings import MatchType

@app.post('/ban-substrings-output', tags=["Ban Substrings"])
async def ban_substrings(request: BanSubstringsScanRequest):
    """
    Scans both the input prompt and model output for mentions of banned substrings.
    The function will flag or redact any matches based on user configuration.
    
    Args:
        request (BanSubstringsScanRequest): The request body containing prompt, model output, 
                                             substrings list, match type, redact flag, and other settings.
    
    Returns:
        dict: A response with sanitized model output, risk score, and success message.

    Raises:
        HTTPException: If the model output contains any banned substrings.
    """
    prompt = request.prompt
    model_output = request.model_output
    substrings = request.substrings
    match_type = request.match_type
    case_sensitive = request.case_sensitive
    redact = request.redact
    contains_all = request.contains_all

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    # Initialize the BanSubstrings scanner with provided configuration
    scanner = BanSubstrings(
        substrings=substrings,
        match_type=match_type,
        case_sensitive=case_sensitive,
        redact=redact,
        contains_all=contains_all
    )

    # Scan the model output for banned substrings
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains banned substrings")

    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }



# Route to scan for code snippets in the prompt
@app.post("/scan_code_snippets", tags=["Code"])
async def scan_code(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the scanner for a specific language (Python in this case) and set it to block
        scanner = Code(languages=["Python"], is_blocked=True)
        
        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MatchType(str, Enum):
    FULL = "full"  # Check the entire output for gibberish
    PARTIAL = "partial"  # Check only certain parts of the output

class GibberishScanRequest(BaseModel):
    """
    Request model for scanning the output for gibberish content.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
        match_type (MatchType): The granularity of the matching - either 'full' or 'partial'.
    """
    prompt: str
    model_output: str
    match_type: MatchType = MatchType.FULL  # Default to full text matching


# Route to scan for gibberish in the prompt
@app.post("/scan_gibberish", tags=["Gibberish"])
async def scan_gibberish(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Gibberish scanner with full match type
        scanner = Gibberish(match_type=MatchType.FULL)
        
        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/gibberish-scanner-output', tags=["Gibberish"])
async def gibberish_scanner(request: GibberishScanRequest):
    """
    Scans the input prompt and model output for gibberish content.
    
    Args:
        request (GibberishScanRequest): The request body containing the prompt, model output, and match type.
    
    Returns:
        dict: A response with sanitized model output, risk score, and success message.
    
    Raises:
        HTTPException: If the model output contains gibberish content.
    """
    prompt = request.prompt
    model_output = request.model_output

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    # Force the match_type to FULL, ignoring the request value
    scanner = Gibberish(match_type=MatchType.FULL)

    # Scan the model output for gibberish content
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains gibberish content")

    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }




# Route to scan for invisible text in the prompt
@app.post("/scan_invisible_text", tags=["Invisible Text"])
async def scan_invisible_text(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the InvisibleText scanner
        scanner = InvisibleText()

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LanguageSameScanRequest(BaseModel):
    """
    Request model for checking if the prompt and output are in the same language.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
    """
    prompt: str
    model_output: str

# Route to scan if prompt and model output are in the same language
@app.post("/scan_language_same_output", tags=["Language"])
async def scan_language_same(request: LanguageSameScanRequest):
    """
    Scans the input prompt and model output to check if they are in the same language.
    
    Args:
        request (LanguageSameScanRequest): The request body containing the prompt and model output.
    
    Returns:
        dict: A response with status, sanitized output, and risk score.
    
    Raises:
        HTTPException: If the prompt and model output are in different languages.
    """
    prompt = request.prompt
    model_output = request.model_output

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    try:
        # Initialize the LanguageSame scanner
        scanner = LanguageSame()
        
        # Perform the language check
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

        if not is_valid:
            raise HTTPException(status_code=400, detail="Prompt and model output are in different languages")
        
        return {
            "sanitized_output": sanitized_output,
            "message": "Prompt and model output are in the same language",
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Route to scan the language in the prompt
@app.post("/scan_language", tags=["Language"])
async def scan_language(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Language scanner with supported languages (ISO 639-1 codes)
        valid_languages = ["en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh", "ar", "hi", "pl", "ru", "sw", "th", "tr", "ur", "vi"]
        scanner = Language(valid_languages=valid_languages, match_type=MatchType.FULL)

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to scan the prompt for injection attempts
@app.post("/scan_prompt_injection", tags=["Prompt Injection"])
async def scan_prompt_injection(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the PromptInjection scanner with a threshold and match type
        scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to scan a prompt for regex patterns
@app.post("/scan_regex", tags=["Regex"])
async def scan_regex(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Define the regex pattern to match (e.g., Bearer token pattern)
        scanner = Regex(
            patterns=[r"Bearer [A-Za-z0-9-._~+/]+"],  # Example pattern (Bearer token)
            is_blocked=True,  # Block any pattern match as 'bad'
            match_type=MatchType.SEARCH,  # Match any part of the prompt
            redact=True,  # Enable redaction
        )

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RegexScanRequest(BaseModel):
    """
    Request model for scanning the output based on regex patterns.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
        patterns (list): List of regex patterns to match.
        is_blocked (bool): Whether matching patterns should mark the output as invalid.
        match_type (str): Match type, either 'SEARCH' or 'FULL_MATCH'.
        redact (bool): Whether to redact the matched portions in the output.
    """
    prompt: str
    model_output: str
    patterns: list
    is_blocked: bool = True  # Default to blocking if pattern is found
    match_type: str = "SEARCH"  # Can be 'SEARCH' or 'FULL_MATCH'
    redact: bool = True  # Enable redaction of matched patterns

# Route to scan model output based on regex patterns
@app.post("/scan_regex_output", tags=["Regex"])
async def scan_regex(request: RegexScanRequest):
    """
    Scans the model output based on predefined regex patterns.
    
    Args:
        request (RegexScanRequest): The request body containing the prompt, model output, regex patterns, and other settings.
    
    Returns:
        dict: A response with sanitized output, risk score, and success message.
    
    Raises:
        HTTPException: If any invalid content is found based on the is_blocked flag.
    """
    prompt = request.prompt
    model_output = request.model_output
    patterns = request.patterns
    is_blocked = request.is_blocked
    match_type = request.match_type
    redact = request.redact

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    try:
        # Initialize the Regex scanner with provided parameters
        scanner = Regex(
            patterns=patterns,
            is_blocked=is_blocked,
            match_type=MatchType[match_type],  # Converting match_type string to MatchType enum
            redact=redact
        )
        
        # Perform the scan
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

        if not is_valid:
            raise HTTPException(status_code=400, detail="Output contains blocked content")

        return {
            "sanitized_output": sanitized_output,
            "message": "Output processed successfully",
            "risk_score": risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Route to scan a prompt for secrets
@app.post("/scan_secrets", tags=["Secrets"])
async def scan_secrets(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Secrets scanner with redaction mode set to partial
        scanner = Secrets(redact_mode=Secrets.REDACT_PARTIAL)

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route to scan a prompt for sentiment
@app.post("/scan_sentiment", tags=["Sentiment"])
async def scan_sentiment(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Sentiment scanner with a threshold for negative sentiment
        scanner = Sentiment(threshold=0)  # You can adjust the threshold as needed

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SentimentScanRequest(BaseModel):
    """
    Request model for scanning the output based on sentiment analysis.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
        threshold (float): The sentiment threshold for flagging negative sentiment.
    """
    prompt: str
    model_output: str
    threshold: float = 0  # Default to 0, meaning neutral or negative sentiments will be flagged

# Route to scan model output based on sentiment analysis
@app.post("/scan_sentiment_output", tags=["Sentiment"])
async def scan_sentiment(request: SentimentScanRequest):
    """
    Scans the model output for sentiment based on the SentimentIntensityAnalyzer.
    
    Args:
        request (SentimentScanRequest): The request body containing the prompt, model output, and sentiment threshold.
    
    Returns:
        dict: A response with sanitized output, sentiment risk score, and success message.
    
    Raises:
        HTTPException: If the sentiment score falls below the threshold, indicating negative sentiment.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    try:
        # Initialize SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        
        # Calculate sentiment score for the model output
        sentiment_score = sia.polarity_scores(model_output)['compound']

        # If sentiment score is below the threshold, mark it as invalid (negative sentiment)
        if sentiment_score < threshold:
            raise HTTPException(status_code=400, detail="Output has negative sentiment")

        return {
            "sanitized_output": model_output,  # Here, you can also redact/modify if needed
            "message": "Sentiment analysis passed successfully",
            "sentiment_score": sentiment_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Route to scan a prompt for token limit
@app.post("/scan_token_limit", tags=["Token Limit"])
async def scan_token_limit(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Token Limit scanner with a predefined token limit (e.g., 4096 tokens)
        scanner = TokenLimit(limit=4096, encoding_name="cl100k_base")

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ToxicityScanRequest(BaseModel):
    """
    Request model for scanning the output based on toxicity detection.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
        threshold (float): The toxicity threshold for flagging toxic output.
        match_type (MatchType): The granularity of the matching, either full text or sentence level.
    """
    prompt: str
    model_output: str
    threshold: float = 0.5  # Default threshold for toxicity score
    match_type: MatchType = MatchType.FULL  # Default to sentence-level matching


# Route to scan model output for toxicity detection
@app.post("/scan_toxicity_output", tags=["Toxicity"])
async def scan_toxicity(request: ToxicityScanRequest):
    """
    Scans the model output for toxicity using the Hugging Face 'unitary/unbiased-toxic-roberta' model.
    
    Args:
        request (ToxicityScanRequest): The request body containing the prompt, model output, toxicity threshold, and match type.
    
    Returns:
        dict: A response with sanitized output, toxicity risk score, and success message.
    
    Raises:
        HTTPException: If the output is classified as toxic based on the threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold
    match_type = request.match_type

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    try:
        # Initialize the toxicity detection pipeline
        toxicity_model = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

        # Use different matching logic based on match_type (sentence-level or full text)
        if match_type == MatchType.FULL:
            sentences = model_output.split(".")
            for sentence in sentences:
                toxicity_result = toxicity_model(sentence)
                toxicity_score = toxicity_result[0]['score']
                if toxicity_score > threshold:
                    raise HTTPException(status_code=400, detail="Toxic content detected in sentence: {}".format(sentence))
        
        elif match_type == MatchType.FULL:
            toxicity_result = toxicity_model(model_output)
            toxicity_score = toxicity_result[0]['score']
            if toxicity_score > threshold:
                raise HTTPException(status_code=400, detail="Toxic content detected in the output")

        return {
            "sanitized_output": model_output,  # The output is returned sanitized (here no modification is made)
            "message": "Output passed toxicity scan successfully",
            "toxicity_score": toxicity_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Route to scan a prompt for toxicity
@app.post("/scan_toxicity", tags=["Toxicity"])
async def scan_toxicity(request: BaseModel):
    prompt = request.dict().get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the Toxicity scanner with a threshold of 0.5
        scanner = Toxicity(threshold=0.5, match_type=MatchType.SENTENCE)

        # Perform the scan
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BiasScanRequest(BaseModel):
    """
    Request model for scanning the output for potential bias detection.
    
    Attributes:
        prompt (str): The input prompt for the language model.
        model_output (str): The output generated by the language model.
        threshold (float): The bias threshold for flagging biased output.
        match_type (MatchType): The granularity of the matching, either full text or sentence level.
    """
    prompt: str
    model_output: str
    threshold: float = 0.5  # Default threshold for bias score
    match_type: MatchType = MatchType.FULL  # Default to full text matching


@app.post("/scan_bias_output", tags=["Bias"])
async def scan_bias(request: BiasScanRequest):
    """
    Scans the model output for bias detection using the Hugging Face 'valurank/distilroberta-bias' model.
    
    Args:
        request (BiasScanRequest): The request body containing the prompt, model output, bias threshold, and match type.
    
    Returns:
        dict: A response with sanitized output, bias risk score, and success message.
    
    Raises:
        HTTPException: If the output is classified as biased based on the threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold
    match_type = request.match_type or MatchType.FULL  # Default to FULL if match_type is not provided

    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")

    try:
        # Initialize the bias detection scanner
        scanner = Bias(threshold=threshold, match_type=match_type)

        # Perform bias scan
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)

        if not is_valid:
            raise HTTPException(status_code=400, detail="Biased content detected in the output")

        return {
            "sanitized_output": sanitized_output,
            "message": "Output passed bias scan successfully",
            "bias_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for input data
class MaliciousScanRequest(BaseModel):
    prompt: str
    model_output: str
    threshold: float = 0.7  # Default threshold for malicious URL detection

@app.post("/scan_malicious_urls", tags=["Malicious URL"])
async def scan_malicious_urls(request: MaliciousScanRequest):
    """
    Scans the model output for malicious URLs using the Hugging Face 'DunnBC22/codebert-base-Malicious_URLs' model.
    
    Args:
        request (MaliciousScanRequest): The request body containing the prompt, model output, and threshold.
    
    Returns:
        dict: A response with sanitized output, validity status, and risk score.
    
    Raises:
        HTTPException: If any malicious URLs are detected based on the threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold
    
    try:
        # Initialize the malicious URL scanner
        scanner = MaliciousURLs(threshold=threshold)
        
        # Perform the scan
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        
        # If a malicious URL is detected, raise an exception
        if not is_valid:
            raise HTTPException(status_code=400, detail="Malicious URL detected in the output")
        
        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed malicious URL scan successfully",
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for input data
class NoRefusalScanRequest(BaseModel):
    prompt: str
    model_output: str
    threshold: float = 0.5  # Default threshold for refusal detection
    match_type: MatchType = MatchType.FULL  # Default match type

@app.post("/scan_no_refusal", tags=["No Refusal"])
async def scan_no_refusal(request: NoRefusalScanRequest):
    """
    Scans the model output for refusals using the ProtectAI/distilroberta-base-rejection-v1 model or a lighter rule-based approach.
    
    Args:
        request (NoRefusalScanRequest): The request body containing the prompt, model output, threshold, and match type.
    
    Returns:
        dict: A response with sanitized output, validity status, and risk score.
    
    Raises:
        HTTPException: If any refusal is detected based on the threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold
    match_type = request.match_type

    try:
        # Initialize the NoRefusal scanner based on match type
        if match_type == MatchType.FULL:
            scanner = NoRefusal(threshold=threshold, match_type=match_type)
        else:
            # Use the lighter version of the NoRefusal scanner if needed
            scanner = NoRefusalLight()

        # Perform the scan
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        
        # If a refusal is detected, raise an exception
        if not is_valid:
            raise HTTPException(status_code=400, detail="Refusal detected in the output")

        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed refusal scan successfully",
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for input data
class FactualConsistencyScanRequest(BaseModel):
    prompt: str
    model_output: str
    minimum_score: float = 0.7  # Minimum score threshold for factual consistency

@app.post("/scan_factual_consistency", tags=["Factual Consistency"])
async def scan_factual_consistency(request: FactualConsistencyScanRequest):
    """
    Scans the model output for factual consistency with the given prompt.
    It leverages a pretrained NLI model to assess contradictions or conflicts between the prompt and model output.
    
    Args:
        request (FactualConsistencyScanRequest): The request body containing the prompt, model output, and minimum score.
    
    Returns:
        dict: A response with sanitized output, validity status, and risk score.
    
    Raises:
        HTTPException: If the output contradicts the prompt based on the configured threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    minimum_score = request.minimum_score

    try:
        # Initialize the FactualConsistency scanner
        scanner = FactualConsistency(minimum_score=minimum_score)

        # Perform the scan to check for factual consistency
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        
        # If the model output is inconsistent with the prompt, raise an exception
        if not is_valid:
            raise HTTPException(status_code=400, detail="Contradiction detected in the output")

        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed factual consistency check",
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for input data
class RelevanceScanRequest(BaseModel):
    prompt: str
    model_output: str
    threshold: float = 0.5  # Default threshold for relevance check

@app.post("/scan_relevance", tags=["Relevance"])
async def scan_relevance(request: RelevanceScanRequest):
    """
    Scans the model output for relevance with the given prompt.
    It calculates the cosine similarity between the prompt and output to assess their contextual alignment.
    
    Args:
        request (RelevanceScanRequest): The request body containing the prompt, model output, and threshold.
    
    Returns:
        dict: A response with sanitized output, validity status, and risk score.
    
    Raises:
        HTTPException: If the output does not meet the relevance threshold.
    """
    prompt = request.prompt
    model_output = request.model_output
    threshold = request.threshold

    try:
        # Initialize the Relevance scanner
        scanner = Relevance(threshold=threshold)

        # Perform the scan to check for relevance
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        
        # If the output is not relevant based on the threshold, raise an exception
        if not is_valid:
            raise HTTPException(status_code=400, detail="Irrelevant content detected in the output")

        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed relevance check",
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for input data
class SensitiveScanRequest(BaseModel):
    prompt: str
    model_output: str
    entity_types: list = ["PERSON", "EMAIL"]  # Default sensitive entity types to scan
    redact: bool = True  # Whether to redact sensitive information

@app.post("/scan_sensitive", tags=["Sensitivity"])
async def scan_sensitive(request: SensitiveScanRequest):
    """
    Scans the model output for sensitive information (like PII), such as names and emails.
    If sensitive information is detected, it can be redacted or flagged.

    Args:
        request (SensitiveScanRequest): The request body containing the prompt, model output, entity types, and redaction flag.
    
    Returns:
        dict: A response with sanitized output, validity status, and risk score.
    
    Raises:
        HTTPException: If sensitive information is detected and redaction fails or if any error occurs.
    """
    prompt = request.prompt
    model_output = request.model_output
    entity_types = request.entity_types
    redact = request.redact

    try:
        # Initialize the Sensitive scanner
        scanner = Sensitive(entity_types=entity_types, redact=redact)

        # Perform the scan to check for sensitive data
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        
        # If invalid (i.e., sensitive information was not successfully redacted or flagged)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Sensitive information detected in the output")

        return {
            "sanitized_output": sanitized_output,  # The output after redaction of sensitive data
            "message": "Output passed sensitive data check",
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Root endpoint
@app.get("/", tags=["Main"])
def read_root():
    return {"name": "LLM Guard API"}

# Health check endpoint
@app.get("/healthz", tags=["Health"])
def read_healthcheck():
    return {"status": "alive"}