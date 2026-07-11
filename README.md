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

Interface web disponível em: [finlab-front](https://github.com/priscilasanfer/finlab-front)

---

## Branch `extras` — Otimizações para Produção

A branch [`extras`](https://github.com/priscilasanfer/finlab/tree/extras) contém duas melhorias pensadas para uso em produção: troca do modelo de embedding e compressão de vetores com quantização.

### 1. Modelo de Embedding Multilingual

O modelo padrão do curso (`all-MiniLM-L6-v2`) foi substituído pelo [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large).

**Por que trocar?**

O `all-MiniLM-L6-v2` tem dois problemas para uso em produção:

- **Limite de tokens baixo:** foi treinado com sequências de 128 tokens e a documentação oficial recomenda truncar em 256. Forçar 300 ou 512 tokens faz o modelo processar textos maiores do que viu no treinamento, piorando a qualidade dos embeddings e aumentando o tempo de processamento.
- **Somente inglês:** não performa bem com queries em português, espanhol ou outros idiomas.

**Vantagens do novo modelo:**

| | `all-MiniLM-L6-v2` | `multilingual-e5-large` |
|---|---|---|
| Parâmetros | 22 milhões | 560 milhões |
| Dimensões do vetor | 384 | 1024 |
| Limite de tokens | 256 (recomendado) | 512 |
| Idiomas | Inglês | 90+ idiomas |

O modelo maior é mais lento na ingestão, mas isso raramente é problema em produção, pois a ingestão acontece offline e o gargalo real nas queries é a chamada ao LLM, não o modelo de embedding.

**Alterações necessárias ao trocar o modelo:**
- Atualizar `dense_model` nos scripts de ingestão e nos chunkers
- Ajustar `max_tokens` para 500
- **Recriar a collection no Qdrant**, pois vetores de 384 dimensões não podem ser misturados com vetores de 1024 dimensões na mesma collection

---

### 2. Quantização — Comprimindo Vetores para Produção

Com vetores de 1024 dimensões em `float32`, cada embedding ocupa 4 KB de RAM (1024 × 4 bytes). Isso escala rapidamente:

| Vetores | RAM necessária (float32) |
|---|---|
| 1 milhão | ~4 GB |
| 10 milhões | ~40 GB |
| 100 milhões | ~400 GB |

A **scalar quantization** converte cada `float32` (4 bytes) em um `int8` (1 byte), reduzindo o tamanho dos vetores em 75% sem perda significativa de qualidade.

**Como funciona:** o Qdrant analisa a distribuição dos seus vetores, descobre o range real que os valores ocupam e mapeia esse range para o intervalo de um `int8` (-128 a 127). A transformação é parcialmente reversível, com perda mínima de precisão.

**Benefícios:**
- **4x menos RAM**: vetores de 1024 dimensões passam de 4 KB para 1 KB
- **Buscas 30–60% mais rápidas**: CPUs modernas executam operações com `int8` mais eficientemente que `float32`

**Configuração no Qdrant (`create_collection`):**

```python
# Vetores originais (float32) ficam em disco — economiza RAM
vectors_config=models.VectorParams(size=1024, on_disk=True, ...),

# Vetores quantizados (int8) ficam em RAM — buscas rápidas
quantization_config=models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,   # ignora 1% de outliers no cálculo do range
        always_ram=True, # mantém os quantizados sempre em RAM
    )
)
```

O parâmetro `quantile=0.99` faz o Qdrant ignorar o 1% de valores mais extremos ao calcular o range de mapeamento, evitando que outliers distorçam a precisão para os 99% dos casos normais.

**Rescoring (opcional):** como a quantização introduz uma pequena imprecisão na ordenação, o Qdrant suporta rescoring — ele busca mais candidatos usando os vetores quantizados em RAM e depois reordena com os vetores originais em `float32` do disco:

```python
search_params=models.SearchParams(
    quantization=models.QuantizationSearchParams(
        ignore=False,
        rescore=True,
        oversampling=1.5,  # busca 1.5x mais candidatos antes de reordenar
    )
)
```

O `oversampling` controla o tradeoff: valores maiores aumentam a precisão mas também a latência. Um valor entre 1.5 e 2.0 é um bom ponto de partida.

> **Quando usar quantização:** compensa a partir de ~1 milhão de vetores. Para bases menores (~500 mil vetores), a complexidade adicional normalmente não justifica os benefícios. Modelos de alta dimensionalidade (768, 1024, 1536+ dimensões) se beneficiam proporcionalmente mais.
