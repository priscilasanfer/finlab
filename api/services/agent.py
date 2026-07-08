import asyncio

from groq import AsyncGroq

from api.config.prompts import (
    AGGREGATION_PROMPT,
    FUNDAMENTAL_PROMPT,
    FUNDAMENTAL_QUERIES,
    MOMENTUM_PROMPT,
    MOMENTUM_QUERIES,
    SENTIMENT_PROMPT,
    SENTIMENT_QUERY_TEMPLATE,
)
from api.config.settings import settings
from api.models.agent import AgentResponse
from api.services.search import SearchService


class AgentService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    def run_queries(self, queries: list[str], limit: int) -> str:
        all_results = []
        for query in queries:
            search_results = self.search_service.search(query, limit)
            all_results.extend([result.text for result in search_results.results])

        return "\n\n".join(all_results)

    async def _generate_completion(self, prompt: str):
        response = await self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message

    async def _analyze_fundamental(self, limit: int):
        context = self.run_queries(FUNDAMENTAL_QUERIES, limit)
        prompt = FUNDAMENTAL_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_momentum(self, limit: int):
        context = self.run_queries(MOMENTUM_QUERIES, limit)
        prompt = MOMENTUM_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_sentiment(self, ticker: str, limit: int):
        query = SENTIMENT_QUERY_TEMPLATE.format(ticker=ticker)
        results = self.search_service.search(query, limit)
        context = "\n\n".join([result.text for result in results.results])
        prompt = SENTIMENT_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def analyze(self, ticker: str, limit: int = 3) -> AgentResponse:
        fundamental_task = self._analyze_fundamental(limit)
        momentum_task = self._analyze_momentum(limit)
        sentiment_task = self._analyze_sentiment(ticker, limit)

        (
            fundamental_analysis,
            momentum_analysis,
            sentiment_analysis,
        ) = await asyncio.gather(fundamental_task, momentum_task, sentiment_task)

        aggregation_prompt = AGGREGATION_PROMPT.format(
            fundamental=fundamental_analysis,
            momentum=momentum_analysis,
            sentiment=sentiment_analysis,
        )
        final_recommendation = await self._generate_completion(aggregation_prompt)

        return AgentResponse(
            ticker=ticker,
            fundamental_analysis=fundamental_analysis,
            momentum_analysis=momentum_analysis,
            sentiment_analysis=sentiment_analysis,
            final_recommendation=final_recommendation,
        )
