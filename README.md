# Tema 03 — Detectores de Bordas por Gradiente (Prewitt, Sobel e Scharr)

Trabalho da disciplina de **Computação Gráfica** apresentando a fundamentação teórica, implementação prática em Python e análise de parâmetros dos operadores de detecção de bordas por gradiente: **Prewitt**, **Sobel** e **Scharr**.

**Integrantes:** Eduardo Ribeiro · Vitor Paloco · Diego Vinicius

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Diego522-dev/computacao_grafica_bordas_gradiente/blob/main/notebooks/tema03_bordas_gradiente.ipynb)

---

## Estrutura do Repositório

```text
.
├── notebooks/
│   └── tema03_bordas_gradiente.ipynb   # Notebook interativo pronto para o Google Colab
├── src/
│   └── detectores_bordas.py            # Script Python standalone com todos os experimentos
├── requirements.txt                    # Dependências Python
└── README.md
```

## O que é demonstrado no código

1. **Pipeline completo (Sobel 3×3):** `Original → |Gx| (bordas verticais) → |Gy| (bordas horizontais) → Magnitude |∇f| → Binarizada (p90)`.
2. **Comparação dos 3 operadores (3×3):** Prewitt (máscara explícita via `cv2.filter2D`), Sobel (`cv2.Sobel`) e Scharr (`cv2.Scharr`).
3. **Experimento 1 — Diferença de escala:** efeito do mesmo limiar fixo (`T = 150`) sobre Prewitt, Sobel e Scharr.
4. **Experimento 2 — Tamanho do kernel (`ksize`):** Sobel `3×3`, `5×5` e `7×7`.
5. **Experimento 3 — Ruído e pré-suavização:** efeito do ruído Gaussiano sem filtro vs `GaussianBlur` (`3×3`, `5×5` e `7×7`).
6. **Experimento 4 — Magnitude `L2` vs `L1`:** comparação entre norma Euclidiana `√(Gx² + Gy²)` e soma dos módulos `|Gx| + |Gy|`.

## Como executar localmente

```bash
pip install -r requirements.txt
python src/detectores_bordas.py
```

Ou clique no botão **Abrir no Google Colab** no topo deste README para executar direto no navegador.
