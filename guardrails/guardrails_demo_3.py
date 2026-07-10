import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
from groq import Groq
from guardrails.validators import FailResult, PassResult, register_validator

from guardrails import Guard

load_dotenv()


client = Groq()


def groq_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )

    return response.choices[0].message.content


@register_validator(name="groq_topic_check", data_type="string")
def groq_topic_check(value, metadata):
    prompt = f"""Is this query about financial analysis or stock? Answer only YES or NO
Query: {value}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
    )

    if "YES" in response.choices[0].message.content.upper():
        return PassResult()
    else:
        return FailResult(errorMessage="Query is not about financial analysis or stock")


guard = Guard().use(groq_topic_check(on_fail="exception"))


query = [
    "How is apple stock doing?",
    "What's the weather today?",
]


for query in query:
    print(f"\nQuery: {query}")
    try:
        guard.validate(query)
        result = groq_wrapper(messages=[{"role": "user", "content": query}])
        print(result)
    except Exception as e:
        print(f"BLOCKED: {e}")
