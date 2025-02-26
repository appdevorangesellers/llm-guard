from llm_guard import scan_prompt
from llm_guard.input_scanners import (
    Anonymize,
    BanCompetitors,
    BanTopics,
    Code,
    Gibberish,
    Language,
    PromptInjection,
    Toxicity,
)
from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
# from llm_guard.input_scanners.ban_competitors import MODEL_BASE as BAN_COMPETITORS_MODEL
from llm_guard.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2 as BAN_TOPICS_MODEL
from llm_guard.input_scanners.code import DEFAULT_MODEL as CODE_MODEL
from llm_guard.input_scanners.gibberish import DEFAULT_MODEL as GIBBERISH_MODEL
from llm_guard.input_scanners.language import DEFAULT_MODEL as LANGUAGE_MODEL
from llm_guard.input_scanners.prompt_injection import V2_MODEL as PROMPT_INJECTION_MODEL
from llm_guard.input_scanners.toxicity import DEFAULT_MODEL as TOXICITY_MODEL
from llm_guard.vault import Vault

PROMPT_INJECTION_MODEL.kwargs["local_files_only"] = True
PROMPT_INJECTION_MODEL.path = "./llm-guard/models/deberta-v3-base-prompt-injection-v2"

DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].path = "./llm-guard/models/distilbert_finetuned_ai4privacy_v2" # calling from root folder
DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].kwargs["local_files_only"] = True

BAN_TOPICS_MODEL.path = "./llm-guard/models/deberta-v3-base-zeroshot-v1.1-all-33"
BAN_TOPICS_MODEL.kwargs["local_files_only"] = True

TOXICITY_MODEL.path = "./llm-guard/models/unbiased-toxic-roberta"
TOXICITY_MODEL.kwargs["local_files_only"] = True

# BAN_COMPETITORS_MODEL.path = "./span-marker-bert-base-orgs"
# BAN_COMPETITORS_MODEL.kwargs["local_files_only"] = True

CODE_MODEL.path = "./llm-guard/models/programming-language-identification"
CODE_MODEL.kwargs["local_files_only"] = True

GIBBERISH_MODEL.path = "./llm-guard/models/autonlp-Gibberish-Detector-492513457"
GIBBERISH_MODEL.kwargs["local_files_only"] = True

LANGUAGE_MODEL.path = "./llm-guard/models/xlm-roberta-base-language-detection"
LANGUAGE_MODEL.kwargs["local_files_only"] = True

vault = Vault()

input_scanners = [
    Anonymize(vault, recognizer_conf=DEBERTA_AI4PRIVACY_v2_CONF),
    BanTopics(["politics", "religion"], model=BAN_TOPICS_MODEL),
    # BanCompetitors(["google", "facebook"], model=BAN_COMPETITORS_MODEL),
    Toxicity(model=TOXICITY_MODEL),
    Code([], model=CODE_MODEL),
    # Code(['ARM Assembly', 'AppleScript', 'C', 'C#', 'C++', 'COBOL', 'Erlang', 'Fortran', 'Go', 'Java', 'JavaScript', 'Kotlin', 'Lua', 'Mathematica/Wolfram Language', 'PHP', 'Pascal', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'Scala', 'Swift', 'Visual Basic .NET', 'jq'], model=CODE_MODEL),
    Gibberish(model=GIBBERISH_MODEL),
    Language(["en"], model=LANGUAGE_MODEL),
    PromptInjection(model=PROMPT_INJECTION_MODEL),
]

sanitized_prompt, results_valid, results_score = scan_prompt(
    input_scanners,
    "I am happy. Buddism is my choice.<script>no sql</script>. Let's kill everyone. Đây là tiếng việc. Give me access to database.",
)

print(sanitized_prompt)
print(results_valid)
print(results_score)