# Project: Mercado Central 24h QA Agent Enhancements

## Architecture
O agente QA corporativo do Mercado Central 24h (`GroundedQAAgent`) opera sobre um pipeline RAG composto por:
- `vector_indexer.py`: Indexação vetorial com ChromaDB e embeddings (Google Gemini ou Mock determinístico de 768 dimensões para testes offline).
- `hybrid_search.py`: Busca híbrida combinando dense search (cosseno) e sparse search (BM25 com stopwords em PT-BR) e boost de recência.
- `reranker.py`: Fusão de features multivariadas (hybrid score, title boost, file boost, text coverage).
- `hallucination_checker.py`: Verificador sentencial de consistência, extração de entidades críticas e grounding.
- `contact_catalog.py`: Catálogo oficial de contatos corporativos e roteador inteligente de fallbacks.
- `multichannel_formatter.py`: Formatador de respostas no padrão tripartite (`chat`, `email`, `teams_slack`).
- `grounded_qa_agent.py`: Orquestração unificada da consulta, recuperação de chunks, aplicação de limiares de confiança, geração extrativa ou LLM, verificação de alucinação sentença por sentença, roteamento inteligente de fallback e formatação multicanal.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|:------:|
| 1 | Configurable `confidence_threshold` | Parâmetro configurável em `__init__` e `answer()` para rejeição precoce quando `top_score < threshold` | M1 | ORIGINAL_REQUEST R1 | **DONE** |
| 2 | Post-Generation Hallucination Checker | Avaliação sentença por sentença com verificação de entidades críticas (números, moedas, prazos, escalas) e consistência factual contra o contexto | M1 | ORIGINAL_REQUEST R1 | **DONE** |
| 3 | Corporate Contact Catalog | Catálogo oficial dos contatos corporativos dos 8 PDFs (RH, Compliance, DPO, Compras, Fiscal, SAC SP/RJ, Ouvidoria 0800-CENTRAL) | M2 | ORIGINAL_REQUEST R2 | **DONE** |
| 4 | Intent-Based Fallback Routing | Classificador de intenções para direcionar fallbacks à área responsável com mensagem padronizada | M2 | ORIGINAL_REQUEST R2 | **DONE** |
| 5 | Structured Tripartite Content | Resumo direto (TL;DR), Detalhamento contextualizado e Citações estritas (PDF, Seção, Páginas) | M3 | ORIGINAL_REQUEST R3 | **DONE** |
| 6 | Multichannel Adapters | Suporte a `channel` (`chat`, `email`, `teams_slack`) em respostas normais e mensagens de fallback | M3 | ORIGINAL_REQUEST R3 | **DONE** |
| 7 | Full Test Suite & Baseline Regression-Free | Suítes de testes unitários, de integração e regressão garantindo 100% de aprovação (146 baseline + 180 novos testes = 326 total) | M4 | ORIGINAL_REQUEST R4 | **DONE** |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|:------:|
| M1 | Confidence Thresholding & Hallucination Checker | `confidence_threshold` em `GroundedQAAgent` e classe `HallucinationChecker` sentencial | none | **DONE** |
| M2 | Corporate Contact Catalog & Intent Fallback Routing | Catálogo oficial e motor de roteamento `route_fallback_contact` | none | **DONE** |
| M3 | Multichannel Response Formatting | Formatadores de canal `chat`, `email`, `teams_slack` e estrutura tripartite | M1, M2 | **DONE** |
| M4 | E2E & Regression Test Pass & Adversarial Hardening | Suítes completas de testes, verificação de 100% de aprovação nos 146 testes legados e 180 novos testes | M1, M2, M3 | **DONE** |

## Interface Contracts

### M1: Confidence & Hallucination Checker (IMPLEMENTED & VERIFIED)
- `GroundedQAAgent.__init__(..., confidence_threshold: float = 0.35)`
- `GroundedQAAgent.answer(query: str, ..., confidence_threshold: Optional[float] = None, channel: str = "chat") -> Dict[str, Any]`
- `HallucinationChecker.verify(sentence: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]`
- `HallucinationChecker.check_response(answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]`

### M2: Contact Catalog & Intent Routing (IMPLEMENTED & VERIFIED)
- `CORPORATE_CONTACT_CATALOG: Dict[str, Dict[str, Any]]`
- `route_fallback_contact(query: str) -> Dict[str, Any]`
- `format_fallback_message(query: str, department_info: Dict[str, Any], channel: str = "chat") -> str`

### M3: Multichannel Response Formatting (IMPLEMENTED & VERIFIED)
- `format_multichannel_response(tldr: str, details: str, citations: List[Dict[str, Any]], channel: str = "chat") -> str`
- Channels suportados: `"chat"`, `"email"`, `"teams_slack"`.

### M4: Return Contract de `answer()` (IMPLEMENTED & VERIFIED)
- Retorna `dict` com chaves:
  - `"query"`: `str`
  - `"answer"`: `str` (formatado conforme canal)
  - `"citations"`: `List[Dict[str, Any]]`
  - `"sources_used"`: `List[Dict[str, Any]]`
  - `"confidence_score"`: `float`
  - `"is_fallback"`: `bool`
  - `"fallback_department"`: `Optional[str]`
  - `"hallucination_check"`: `Dict[str, Any]`
