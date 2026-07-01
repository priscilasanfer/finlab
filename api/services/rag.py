from groq import Groq

from api.config.settings import settings
from api.models.rag import RAGResponse
from api.services.search import SearchService


class RAGService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = Groq(api_key=settings.groq_api_key)

    def generate_answers(self, query: str, limit: int = 3):
        search_result = self.search_service.search(query, limit)

        context = "\n\n".join(result.text for result in search_result.results)

        prompt = f"""
        Based on the following financial documents, answer the question.
        Context: {context}
        Question : {query}
        Answer:
        """

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        metadata = [
            {
                **result.metadata,
                "score": result.score,
            }
            for result in search_result.results
        ]

        return RAGResponse(
            query=query,
            answers=response.choices[0].message.content,
            metadata=metadata,
        )
