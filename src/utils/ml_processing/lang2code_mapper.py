import json

def map_language_to_code(language, system="whisper"):
    with open("language_codes.json") as f:
        language2code = json.load(f)["lang2code"]
    print(language2code)

    assert language in language2code, "Language is not corrent."
    
    code = language2code[language][system]
    return code
