"""
Make co-occurrence heatmaps, for a schematic in a publication.
Explains "fragmentation" visually
"""
import numpy as np
import matplotlib.pyplot as plt

from aochildesnouns.figs import plot_heatmap_for_schematic

SIZE = 16

shape = (SIZE, SIZE)

# case 1: no fragmentation

mat = np.ones(shape)
plot_heatmap_for_schematic(mat,
                           title='No fragmentation')

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

# case 4: high fragmentation

mat = np.rot90(np.eye(SIZE))
plot_heatmap_for_schematic(mat,
                           title='Maximal fragmentation')