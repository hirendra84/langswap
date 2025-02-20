import json
import os

def map_language_to_code(language, system="whisper"):
    lang_file = "./src/utils/ml_processing/language_codes.json"
    lang_file = os.path.abspath(lang_file)
    with open(lang_file) as f:
        language2code = json.load(f)["lang2code"]
    assert language in language2code, f"Language {language} is not corrent."
    
    code = language2code[language][system]
    return code
