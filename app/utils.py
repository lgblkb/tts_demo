from num2words import num2words


def resolve_num2words(text: str) -> str:
    return " ".join([
        num2words(word, lang='kz')
        if word.isdigit() else word
        for word in text.split()
    ])
