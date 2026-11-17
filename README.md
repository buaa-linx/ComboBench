# ComboBench

ComboBench studies robust 3D perception under compositional camera–LiDAR corruptions. The project page introduces the configurable ComboGen pipeline, the CARE framework, corrupted observation examples, and experimental results on nuScenes-Combo and KITTI-Combo.

## Project page

[Visit the project page](https://buaa-linx.github.io/ComboBench-pages/)

## Resource status

- Paper PDF: not publicly released yet.
- Code button: [official implementation repository](https://github.com/buaa-linx/ComboBench).
- The Paper button currently links to the official implementation repository.
- Data button: [ComboBench dataset on Hugging Face](https://huggingface.co/datasets/linchunmian/ComboBench).
- Qualitative Results: one shared video player with dataset and corruption-composition selectors; two videos are available for each of nuScenes-combo and KITTI-combo.

## Website source

The page is a static website. Its entry point is `index.html`; all assets use relative paths compatible with a GitHub Pages project site.

Resource links, author URLs, publication metadata, and independent video file URLs are configured in `site-config.js`. Keep unavailable URLs as `null`.

The page presents corruption samples, the CARE architecture, experimental results with the original Figure 5 and Tables 7 and 8, and qualitative detection results.

The complete numerical data from Tables 7 and 8 is also retained in `assets/data/table7-table8.json`. Figure and data provenance is recorded in `SOURCE_NOTES.md`.

## Third-party assets

Icon sources and license notices are preserved in [assets/icons/SOURCES.md](assets/icons/SOURCES.md).

Typography uses locally hosted Noto Sans under the SIL Open Font License; source and license are in `assets/fonts/`.

## Acknowledgements

The typography and video presentation of this project page were inspired by [Academic Project Astro Template](https://github.com/RomanHauksson/academic-project-astro-template) by Roman Hauksson.

## Deployment

This standalone repository serves the project page from the repository root. Set Settings → Pages → Source to Deploy from a branch, branch `main`, folder `/ (root)`.

The four video URLs currently point to the public source repository. Run **Actions → Sync website videos → Run workflow** once to copy the MP4 files locally and change the config to local paths. Do this before making the source repository private.
