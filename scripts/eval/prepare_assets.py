"""Get the pinned benchmark inputs and chat template."""

from hashlib import sha256
from urllib.request import urlopen

from generate import DATA


ASSETS = {
    "ifeval.jsonl": (
        "https://raw.githubusercontent.com/google-research/google-research/ee4298226f21649f45c0d0427f7c0f3f682d2a20/instruction_following_eval/data/input_data.jsonl",
        "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
    ),
    "gsm8k.jsonl": (
        "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    ),
    "olmo3-chat.jinja": (
        "https://huggingface.co/allenai/Olmo-3-7B-Instruct/resolve/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc/chat_template.jinja",
        "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b",
    ),
}


def main():
    DATA.mkdir(exist_ok=True, parents=True)
    for filename, (url, expected) in ASSETS.items():
        path = DATA / filename
        if path.exists():
            actual = sha256(path.read_bytes()).hexdigest()
        else:
            with urlopen(url, timeout=45) as response:
                content = response.read()
            actual = sha256(content).hexdigest()
            if actual == expected:
                path.write_bytes(content)
        if actual != expected:
            raise ValueError(f"{filename} SHA-256 mismatch, expected {expected} and found {actual}")
        print(f"{filename}: {actual}")


if __name__ == "__main__":
    main()
