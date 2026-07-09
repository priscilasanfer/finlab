# FinLab — Financial Intelligence API

API de análise de investimentos com busca semântica e agente multi-stream, construída com FastAPI, Qdrant e Groq (LLaMA 3).

## Visão geral

O FinLab expõe três endpoints principais:

| Endpoint | Descrição |
|----------|-----------|
| `POST /rag` | Busca híbrida (dense + sparse + ColBERT) nos documentos financeiros e gera uma resposta em linguagem natural |
| `POST /agent` | Análise multi-stream paralela: fundamental (10-K), momentum (10-Q) e sentimento (Yahoo Finance), com recomendação final BUY / HOLD / SELL |
| `POST /search` | Busca semântica pura nos documentos indexados no Qdrant |

## Tecnologias

- **Python 3.12+**
- **FastAPI** — framework web
- **Qdrant** — banco vetorial com busca híbrida
- **Groq / LLaMA 3.1** — geração de respostas e análises estruturadas
- **Instructor** — respostas LLM com tipagem via Pydantic
- **FastEmbed** — embeddings dense, sparse (BM25) e ColBERT
- **EdgarTools / yfinance** — coleta de dados financeiros

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Instância do Qdrant (local ou cloud)
- Chave de API da Groq

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=sua_chave_aqui
GROQ_API_KEY=sua_chave_aqui

# Opcionais (já possuem valor padrão)
COLLECTION_NAME=financial
GROQ_MODEL=llama-3.1-8b-instant
```

## Instalação e execução

```bash
# Instalar dependências
uv sync

# Rodar o servidor de desenvolvimento
uv run uvicorn api.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

Documentação interativa: `http://localhost:8000/docs`

## Exemplos de uso

**RAG — pergunta sobre empresas:**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Apple main risks?", "limit": 5}'
```

**Agent — análise completa de um ativo:**
```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "AAPL", "limit": 3}'
```

## Estrutura do projeto

```
api/
├── config/         # Settings, prompts e mapeamentos de empresas
├── models/         # Schemas Pydantic (request/response)
├── routers/        # Endpoints FastAPI
└── services/       # Lógica de negócio (RAG, Agent, Search, Embeddings)
```

## Frontend

Interface web disponível em: [finlab-front](https://github.com/infoslack/finlab-front)
