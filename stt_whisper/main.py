from whisp_test import *
from word_extract import *
import sys

def matching_prompt(prompt_file):
    df = pd.read_csv(prompt_file)
    prompt = transcribe_directly()
    keywords = extract_keywords(prompt)
    print(keywords)
    match = find_prompt_with_words(keywords, df)
    print(match)
    return match if match else "No match"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    matching_prompt(file_path)