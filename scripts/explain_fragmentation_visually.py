"""
Make co-occurrence heatmaps, for a schematic in a publication.
Explains "fragmentation" visually
"""
import numpy as np

from aochildesnouns.figs import plot_heatmap_for_schematic

SIZE = 8

shape = (SIZE, SIZE)


def save_to_text(m: np.array):
    """compatible with matrix plotting in Latex pgfplots"""
    m = m.T
    for j in range(m.shape[1]):
        for i in range(m.shape[0]):
            print(f'{i} {j} {m[i, j]}')
        print()


# case 0: no fragmentation + maximally entropic contexts

mat = np.zeros(shape)
mat[:, 0] = 1
mat[:, 3] = 1
plot_heatmap_for_schematic(mat,
                           title='No fragmentation')
save_to_text(mat)

# case 1: no fragmentation

mat = np.ones(shape)
plot_heatmap_for_schematic(mat,
                           title='No fragmentation')
save_to_text(mat)

# case 2: medium-low fragmentation

a = np.zeros((SIZE // 2, SIZE // 2))
b = np.zeros((SIZE // 2, SIZE // 2)) + 1
mat = np.block(
    [
        [a, b],
        [b, a],
    ]
)
plot_heatmap_for_schematic(np.random.permutation(mat.flatten()).reshape(mat.shape),
                           title='Intermediate-low fragmentation')
save_to_text(mat)

# case 3: medium-high fragmentation

a = np.zeros((SIZE // 2, SIZE // 2))
b = np.zeros((SIZE // 2, SIZE // 2)) + 1
mat = np.block(
    [
        [a, b],
        [b, a],
    ]
)
plot_heatmap_for_schematic(mat,
                           title='Intermediate-high fragmentation')
save_to_text(mat)

# case 4: high fragmentation

mat = np.rot90(np.eye(SIZE))
plot_heatmap_for_schematic(mat,
                           title='Maximal fragmentation')
save_to_text(mat)