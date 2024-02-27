import torch

with open('training_data.txt', 'r', encoding='utf-8') as f:
    text: str = f.read()

chars: list = sorted(list(set(text)))

str_to_int: dict = { char:i for i, char in enumerate(chars) }
int_to_str: dict = { i:char for i, char in enumerate(chars) }

def encode(string: str) -> list:
    return [str_to_int[char] for char in string]

def decode(ints: list) -> str:
    return ''.join([int_to_str[i] for i in ints])