import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
from guardrails.hub import ProfanityFree
from openai import OpenAI

from guardrails import Guard

load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def groq_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )

    return response.choices[0].message.content


guard = Guard().use(ProfanityFree())


# Valida somente a resposta do LLM
validate_response = guard(
    groq_wrapper,
    messages=[
        {"role": "user", "content": "FAANG representa quais empresas de tecnologia?"}
    ],
)

print(validate_response.validated_output)
