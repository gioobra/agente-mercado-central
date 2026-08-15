# E2E Test Infra: Mercado Central 24h QA Agent Enhancements

## Test Philosophy
- Opaque-box & Requirement-driven test suite cobrindo os 4 novos requisitos (R1, R2, R3, R4) e garantindo regressão zero na suíte baseline de 146 testes.
- Hermeticidade total: testes 100% determinísticos e offline utilizando mocks locais sem dependência de rede externa ou chaves de API.

## Feature Inventory & Test Mapping
| # | Feature | Source (Requirement) | Test Suite | Casos Planejados |
|---|---------|----------------------|------------|------------------|
| 1 | Configurable `confidence_threshold` | ORIGINAL_REQUEST R1 | `test_hallucination_and_confidence.py` | >= 6 |
| 2 | Post-Generation Hallucination Checker | ORIGINAL_REQUEST R1 | `test_hallucination_and_confidence.py` | >= 8 |
| 3 | Corporate Contact Catalog (8 PDFs) | ORIGINAL_REQUEST R2 | `test_corporate_routing.py` | >= 8 |
| 4 | Intent-Based Fallback Routing | ORIGINAL_REQUEST R2 | `test_corporate_routing.py` | >= 8 |
| 5 | Multichannel Formatting (Tripartite + Channels) | ORIGINAL_REQUEST R3 | `test_multichannel_formatting.py` | >= 8 |
| 6 | E2E Enhancements & 146 Baseline Compatibility | ORIGINAL_REQUEST R4 | `test_e2e_enhancements.py` + baseline | >= 10 + 146 baseline |

## Test Architecture
- **Runner**: `./venv/bin/pytest rag/tests/ -v`
- **Baseline Files**:
  - `rag/tests/test_pdf_processor.py` (34 testes)
  - `rag/tests/test_rag_pipeline.py` (27 testes)
  - `rag/tests/test_e2e_scenarios.py` (9 testes)
  - `rag/tests/test_adversarial_tier5.py` (76 testes)
- **New Enhancement Test Files**:
  - `rag/tests/test_hallucination_and_confidence.py`
  - `rag/tests/test_corporate_routing.py`
  - `rag/tests/test_multichannel_formatting.py`
  - `rag/tests/test_e2e_enhancements.py`

## Pass/Fail Semantics
- 100% dos testes devem passar (0 failed, 0 errors).
- O tempo total de execução deve ser < 10 segundos.
- Não-regressão estrita: nenhuma asserção de teste legado pode falhar.
