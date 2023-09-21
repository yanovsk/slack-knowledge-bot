import re

import tiktoken


def to_slack_markdown(text):
    text = re.sub(r'#+\s?(.*)', r'*\1*', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<\2|\1>', text)
    text = re.sub(r'\n\s*-\s', '\n• ', text)
    text = re.sub(r'(\d+)\.\s', r'\1) ', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.*?)__', r'_\1_', text)
    for lang in ['python', 'go', 'javascript', 'typescript']:
        text = re.sub(r'```' + lang + r'(.*?)```',
                      r'```\1```',
                      text,
                      flags=re.DOTALL)
    return text


def num_tokens_from_string(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
