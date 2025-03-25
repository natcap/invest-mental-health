def calculate_midpoints(boundaries):
    """
    Calculate the midpoints between each boundary for correct tick placement.
    """
    return [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]


def plot_land_covers_side_by_side(landcover, reclassified, landcover_meta):
    """
    Plot the original and reclassified land cover side by side using NLCD colors.
    """

    # Plotting the original land cover
    plt.figure(figsize=(14, 7))
    
    # Plot 1: Original Land Cover
    plt.subplot(1, 2, 1)
    original_codes = np.unique(landcover)
    original_cmap_colors = np.array([nlcd_colors[code] for code in original_codes]) / 255.0
    original_cmap = ListedColormap(original_cmap_colors)
    original_boundaries = np.concatenate(([original_codes[0] - 0.5], original_codes + 0.5))
    original_norm = BoundaryNorm(original_boundaries, original_cmap.N)
    
    plt.imshow(landcover, cmap=original_cmap, norm=original_norm,
               extent=(landcover_meta['transform'][2], landcover_meta['transform'][2] + landcover_meta['transform'][0] * landcover.shape[1],
                       landcover_meta['transform'][5] + landcover_meta['transform'][4] * landcover.shape[0], landcover_meta['transform'][5]))
    plt.title('Original Land Cover')
    
    original_ticks = calculate_midpoints(original_boundaries)
    plt.colorbar(ticks=original_ticks, format='%d').ax.set_yticklabels([str(int(code)) for code in original_codes])
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Plot 2: Reclassified Land Cover
    plt.subplot(1, 2, 2)
    reclassified_codes = np.unique(reclassified)
    reclassified_cmap_colors = np.array([extended_colors[code] for code in reclassified_codes]) / 255.0
    reclassified_cmap = ListedColormap(reclassified_cmap_colors)
    reclassified_boundaries = np.concatenate(([reclassified_codes[0] - 0.5], reclassified_codes + 0.5))
    reclassified_norm = BoundaryNorm(reclassified_boundaries, reclassified_cmap.N)
    
    plt.imshow(reclassified, cmap=reclassified_cmap, norm=reclassified_norm,
               extent=(landcover_meta['transform'][2], landcover_meta['transform'][2] + landcover_meta['transform'][0] * reclassified.shape[1],
                       landcover_meta['transform'][5] + landcover_meta['transform'][4] * reclassified.shape[0], landcover_meta['transform'][5]))
    plt.title('Reclassified Land Cover')
    
    reclassified_ticks = calculate_midpoints(reclassified_boundaries)
    plt.colorbar(ticks=reclassified_ticks, format='%d').ax.set_yticklabels([str(int(code)) for code in reclassified_codes])
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()

