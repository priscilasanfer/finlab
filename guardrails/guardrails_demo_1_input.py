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


guard = Guard().use(ProfanityFree(on_fail="exception"))


# Valida somente a query que o usuário pergunta ao LLM

query = "FAANG representa quais fucking empresas de tecnologia?"

try:
    guard.validate(query)
except Exception as e:
    print(e)

validate_response = guard(
    groq_wrapper,
    messages=[{"role": "user", "content": query}],
)

print(validate_response.validated_output)
