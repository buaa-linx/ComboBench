# ComboBench

ComboBench studies robust 3D perception under compositional camera–LiDAR corruptions. The project page introduces the configurable ComboGen pipeline, the CARE framework, corrupted observation examples, and experimental results on nuScenes-Combo and KITTI-Combo.

## Project page

[Visit the project page](https://buaa-linx.github.io/ComboBench/)

## Resource status

- Manuscript PDF: not publicly released yet.
- Paper, research code, and data buttons are reserved until public URLs are available.
- nuScenes and KITTI videos: reserved positions.

## Website source

The page is a static website. Its entry point is `index.html`; all assets use relative paths compatible with a GitHub Pages project site.

Resource links, author URLs, publication metadata, and independent video file URLs are configured in `site-config.js`. Keep unavailable URLs as `null`.

The page distinguishes ComboGen (corruption generation and clean-trained robustness evaluation) from CARE (expert adaptation and robustness enhancement). Figure 5 belongs to the evaluation section; the original images of Tables 7 and 8 and Figure 9 belong to the enhancement section.

The complete numerical data from Tables 7 and 8 is also retained in `assets/data/table7-table8.json`. Figure and data provenance is recorded in `SOURCE_NOTES.md`.

## Third-party assets

Icon sources and license notices are preserved in [assets/icons/SOURCES.md](assets/icons/SOURCES.md).
