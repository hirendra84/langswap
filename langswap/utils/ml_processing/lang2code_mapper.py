import json
import os

def map_language_to_code(language, system="whisper"):
    lang_file = "./langswap/utils/ml_processing/language_codes.json"
    lang_file = os.path.abspath(lang_file)
    with open(lang_file) as f:
        language2code = json.load(f)["lang2code"]

    if system == "reverse_from_whisper":
        # Given a whisper code (e.g. "ru"), return the lowercase language name (e.g. "russian")
        for lang_name, codes in language2code.items():
            if codes.get("whisper") == language:
                return lang_name
        return language  # fallback: return as-is

    assert language in language2code, f"Language {language} is not corrent."
    code = language2code[language][system]
    return code
