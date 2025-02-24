from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
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
from llm_guard_api.app.scanner_request import PromptRequest, CompetitorListRequest, BanTopicsRequest, AnonymizeRequest, BanTopicsRequest, BanTopicsOutputRequest, AnonymizationRequest, CodeScanRequest, CompetitorScanRequest, BanSubstringsScanRequest, GibberishScanRequest, LanguageSameScanRequest, RegexScanRequest, SentimentScanRequest, ToxicityScanRequest, BiasScanRequest, MaliciousScanRequest, NoRefusalScanRequest, FactualConsistencyScanRequest, RelevanceScanRequest, SensitiveScanRequest

vault = Vault()
app = FastAPI()
competitor_list = ["A1"]
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

scanner = BanCompetitors(competitors=competitor_list, redact=False, threshold=0.5)
banned_topics = [
    "Violence", "Hate Speech", "Drugs", "Explicit Content", "Terrorism",
    "Pornography", "Child Abuse", "Racism", "Sexism", "Cyberbullying", 
    "Self-Harm", "Discrimination", "Gambling", "Weaponry", "Fraud", 
    "Abortion", "COVID-19 Misinformation", "Animal Cruelty", "Political Extremism", 
    "Extremist Ideologies"
]


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post('/ban-topics', tags=["Ban Topics"])
async def ban_topics(request: BanTopicsRequest):
    """
    Scans the input prompt for banned topics using a zero-shot classifier.
    Returns a success message and risk score if valid.
    """
    prompt = request.prompt
    threshold = request.threshold
    
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    # Create the BanTopics scanner with predefined topics
    banned_topics = ["violence"]  # Define your list of banned topics (this can be adjusted)
    scanner = BanTopics(topics=banned_topics, threshold=threshold)
    
    try:
        # Use the scanner to scan the prompt
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    except Exception as e:
        logger.error(f"Error during scanning: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during topic scan")
    
    if not is_valid:
        logger.error("Prompt contains banned topics")
        raise HTTPException(status_code=400, detail="Prompt contains banned topics")
    
    logger.info("Prompt sanitized successfully")
    return {"message": "Prompt sanitized successfully", "risk_score": risk_score}


# Output Scanner Endpoint
@app.post('/ban-topics-output', tags=["Ban Topics"])
async def ban_topics(request: BanTopicsRequest):
    """
    Scans the input prompt for banned topics using a zero-shot classifier.
    Returns a success message and risk score if valid.
    """
    prompt = request.prompt
    threshold = request.threshold
    
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    # Check for banned topics in the prompt
    found_topics = [topic for topic in banned_topics if topic in prompt]
    
    if found_topics:
        raise HTTPException(status_code=400, detail=f"Prompt contains banned topics: {', '.join(found_topics)}")
    
    # Initialize the scanner (using predefined banned topics and the provided threshold)
    scanner = BanTopics(topics=banned_topics, threshold=threshold)
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Prompt contains banned topics")
    
    return {"message": "Prompt sanitized successfully", "risk_score": risk_score}

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
        request (CodeScanRequest): The request object containing prompt, model output, and a list of banned languages.
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
    scanner = Code(languages=languages, is_blocked=is_blocked)
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains banned code snippets")
    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }

@app.post("/sanitize_competitors", tags=["Ban Competitors"])
async def scan_competitors(request: BaseModel):
    prompt = request.dict().get("prompt")
    output = request.dict().get("output")
    if not prompt or not output:
        raise HTTPException(status_code=400, detail="Prompt and output are required")
    try:
        scanner = BanCompetitors(competitors=competitor_list, redact=False, threshold=0.5)
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
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
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
        request (CompetitorScanRequest): The request body containing prompt, model output, competitors list, redact flag, and threshold.
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
    scanner = BanCompetitors(competitors=competitors, redact=redact, threshold=threshold)
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains competitors' names")
    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }

@app.post("/scan_banned_substrings", tags=["Ban Substrings"])
async def scan_prompt(request: PromptRequest):
    prompt = request.prompt
    try:
        sanitized_prompt, is_valid, risk_score = bansubstr_scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ban-substrings-output', tags=["Ban Substrings"])
async def ban_substrings(request: BanSubstringsScanRequest):
    """
    Scans both the input prompt and model output for mentions of banned substrings.
    The function will flag or redact any matches based on user configuration.
    Args:
        request (BanSubstringsScanRequest): The request body containing prompt, model output, substrings list, match type, redact flag, and other settings.
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
    scanner = BanSubstrings(
        substrings=substrings,
        match_type=match_type,
        case_sensitive=case_sensitive,
        redact=redact,
        contains_all=contains_all
    )
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Model output contains banned substrings")
    return {
        "sanitized_output": sanitized_output,
        "message": "Model output sanitized successfully",
        "risk_score": risk_score
    }

@app.post("/scan_code_snippets", tags=["Code"])
async def scan_code(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Code(languages=["Python"], is_blocked=True)
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_gibberish", tags=["Gibberish"])
async def scan_gibberish(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Gibberish(match_type=MatchType.FULL)
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

@app.post("/scan_invisible_text", tags=["Invisible Text"])
async def scan_invisible_text(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = InvisibleText()
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/scan_language", tags=["Language"])
async def scan_language(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        valid_languages = ["en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh", "ar", "hi", "pl", "ru", "sw", "th", "tr", "ur", "vi"]
        scanner = Language(valid_languages=valid_languages, match_type=MatchType.FULL)
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_prompt_injection", tags=["Prompt Injection"])
async def scan_prompt_injection(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_regex", tags=["Regex"])
async def scan_regex(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Regex(
            patterns=[r"Bearer [A-Za-z0-9-._~+/]+"],  # Example pattern (Bearer token)
            is_blocked=True,  # Block any pattern match as 'bad'
            match_type=MatchType.SEARCH,  # Match any part of the prompt
            redact=True,  # Enable redaction
        )
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/scan_secrets", tags=["Secrets"])
async def scan_secrets(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Secrets(redact_mode=Secrets.REDACT_PARTIAL)
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_sentiment", tags=["Sentiment"])
async def scan_sentiment(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Sentiment(threshold=0)  
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(model_output)['compound']
        if sentiment_score < threshold:
            raise HTTPException(status_code=400, detail="Output has negative sentiment")
        return {
            "sanitized_output": model_output, 
            "message": "Sentiment analysis passed successfully",
            "sentiment_score": sentiment_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_token_limit", tags=["Token Limit"])
async def scan_token_limit(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = TokenLimit(limit=4096, encoding_name="cl100k_base")
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        toxicity_model = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")
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
            "sanitized_output": model_output,  
            "message": "Output passed toxicity scan successfully",
            "toxicity_score": toxicity_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan_toxicity", tags=["Toxicity"])
async def scan_toxicity(request: BaseModel):
    prompt = request.dict().get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        scanner = Toxicity(threshold=0.5, match_type=MatchType.SENTENCE)
        sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
        return {
            "sanitized_prompt": sanitized_prompt,
            "is_valid": is_valid,
            "risk_score": risk_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    match_type = request.match_type or MatchType.FULL 
    if not prompt or not model_output:
        raise HTTPException(status_code=400, detail="Prompt or model output not provided")
    try:
        scanner = Bias(threshold=threshold, match_type=match_type)
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
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Malicious URL detected in the output")
        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed malicious URL scan successfully",
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        if match_type == MatchType.FULL:
            scanner = NoRefusal(threshold=threshold, match_type=match_type)
        else:
            scanner = NoRefusalLight()
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Refusal detected in the output")
        return {
            "sanitized_output": sanitized_output,  
            "message": "Output passed refusal scan successfully",
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        scanner = FactualConsistency(minimum_score=minimum_score)
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Contradiction detected in the output")
        return {
            "sanitized_output": sanitized_output,  
            "message": "Output passed factual consistency check",
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        scanner = Relevance(threshold=threshold)
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Irrelevant content detected in the output")
        return {
            "sanitized_output": sanitized_output,  # The output is returned sanitized
            "message": "Output passed relevance check",
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        scanner = Sensitive(entity_types=entity_types, redact=redact)
        sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Sensitive information detected in the output")
        return {
            "sanitized_output": sanitized_output,  
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