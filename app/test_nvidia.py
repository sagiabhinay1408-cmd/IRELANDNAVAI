
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)


def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    model_name = os.getenv("NVIDIA_MODEL")
    base_url = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1"
    )

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is missing from .env"
        )

    if not model_name:
        raise ValueError(
            "NVIDIA_MODEL is missing from .env"
        )

    if model_name.startswith("http"):
        raise ValueError(
            "NVIDIA_MODEL must be a model ID, not a URL"
        )

    print("=" * 55)
    print("IRELAND NAVIGATOR AI")
    print("NVIDIA Nemotron Connection Test")
    print("=" * 55)

    print(f"\nModel: {model_name}")
    print("Connecting to NVIDIA...")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain retrieval-augmented generation "
                    "in two short sentences."
                ),
            }
        ],
        temperature=0.3,
        top_p=0.95,
        max_tokens=1024,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        stream=False,
    )

    answer = completion.choices[0].message.content

    print("\nMODEL RESPONSE:")
    print(answer or "No text returned by model.")

    print("\nNVIDIA connection test completed!")


if __name__ == "__main__":
    main()
