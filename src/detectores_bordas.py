"""
Tema 03 — Detectores de Bordas por Gradiente (Prewitt, Sobel e Scharr)
Disciplina: Computação Gráfica
Grupo: Eduardo Ribeiro, Vitor Paloco e Diego Vinicius

Script Python standalone que reproduz o pipeline completo de detecção de bordas
por gradiente e os 4 experimentos de análise de parâmetros.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Máscaras 3x3 explícitas do operador de Prewitt (suavização [1, 1, 1] x derivada [-1, 0, +1])
KERNEL_PREWITT_X = np.array(
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]],
    dtype=np.float64,
)
KERNEL_PREWITT_Y = KERNEL_PREWITT_X.T


def detectar_bordas(
    img: np.ndarray,
    operador: str = "sobel",
    ksize: int = 3,
    limiar: float | None = None,
    ruido: bool = False,
    blur: int = 0,
    modo: str = "L2",
):
    """
    Aplica detecção de bordas por gradiente em uma imagem em níveis de cinza.

    Parâmetros:
        img: imagem 2D em níveis de cinza (uint8).
        operador: 'prewitt', 'sobel' ou 'scharr'.
        ksize: tamanho da máscara do Sobel (1, 3, 5 ou 7).
        limiar: valor fixo de corte; se None, usa automaticamente o percentil 90.
        ruido: se True, adiciona ruído Gaussiano (sigma=20) à entrada.
        blur: tamanho do filtro GaussianBlur antes da derivada (0 desativa).
        modo: 'L2' (Euclidiana) ou 'L1' (soma dos módulos).

    Retorna:
        Tupla (gx, gy, mag, binaria, limiar_usado).
    """
    entrada = img.copy()

    # 1) Ruído sintético opcional (semente fixa para reprodutibilidade)
    if ruido:
        np.random.seed(42)
        entrada = np.clip(img + np.random.normal(0, 20, img.shape), 0, 255).astype(np.uint8)

    # 2) Pré-suavização Gaussiana opcional para atenuar ruído antes de derivar
    if blur > 0:
        entrada = cv2.GaussianBlur(entrada, (blur, blur), 0)

    # 3) Derivadas parciais Gx (bordas verticais) e Gy (bordas horizontais)
    # CV_64F preserva valores negativos e magnitudes acima de 255
    if operador == "prewitt":
        gx = cv2.filter2D(entrada.astype(np.float64), -1, KERNEL_PREWITT_X)
        gy = cv2.filter2D(entrada.astype(np.float64), -1, KERNEL_PREWITT_Y)
    elif operador == "scharr":
        gx = cv2.Scharr(entrada, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(entrada, cv2.CV_64F, 0, 1)
    else:
        gx = cv2.Sobel(entrada, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(entrada, cv2.CV_64F, 0, 1, ksize=ksize)

    # 4) Magnitude: L2 (isotrópica) ou L1 (dispensa raiz quadrada, +41% nas diagonais)
    mag = np.sqrt(gx**2 + gy**2) if modo == "L2" else (np.abs(gx) + np.abs(gy))

    # 5) Binarização pelo limiar informado ou percentil 90 automático
    t = float(np.percentile(mag, 90)) if limiar is None else float(limiar)
    binaria = mag >= t
    return gx, gy, mag, binaria, t


def main():
    img = data.camera()

    # --- 1. Pipeline completo (Sobel 3x3) ---
    gx, gy, mag, binaria, t = detectar_bordas(img, operador="sobel", ksize=3)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for ax, (im, titulo) in zip(
        axes,
        [
            (img, "1. Original"),
            (np.abs(gx), "2. |Gx| (verticais)"),
            (np.abs(gy), "3. |Gy| (horizontais)"),
            (mag, "4. Magnitude |∇f|"),
            (binaria, f"5. Binarizada (T={t:.0f})"),
        ],
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 2. Diferença de escala sob o mesmo limiar fixo (T=150) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, op in zip(axes, ["prewitt", "sobel", "scharr"]):
        _, _, mag_op, bin_op, t_op = detectar_bordas(img, operador=op, limiar=150)
        ax.imshow(bin_op, cmap="gray")
        ax.set_title(f"{op.upper()} | máx={mag_op.max():.0f} | T fixo={t_op:.0f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 3. Tamanho do kernel Sobel (3x3, 5x5 e 7x7) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, [3, 5, 7]):
        _, _, mag_k, bin_k, t_k = detectar_bordas(img, operador="sobel", ksize=k, limiar=None)
        ax.imshow(bin_k, cmap="gray")
        ax.set_title(f"Sobel {k}×{k} | máx={mag_k.max():,.0f} | T(p90)={t_k:,.0f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 4. Ruído e pré-suavização Gaussiana ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (titulo, b) in zip(
        axes,
        [
            ("Sem blur (ruído amplificado)", 0),
            ("Gaussiano 3×3", 3),
            ("Gaussiano 5×5 (equilíbrio)", 5),
            ("Gaussiano 7×7 (bordas largas)", 7),
        ],
    ):
        _, _, _, bin_r, _ = detectar_bordas(img, operador="sobel", ruido=True, blur=b, limiar=120)
        ax.imshow(bin_r, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 5. Magnitude L2 vs L1 ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, modo in zip(axes, ["L2", "L1"]):
        _, _, mag_m, bin_m, t_m = detectar_bordas(img, operador="sobel", limiar=150, modo=modo)
        ax.imshow(bin_m, cmap="gray")
        ax.set_title(f"Magnitude {modo} | máx={mag_m.max():.0f} | T={t_m:.0f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
