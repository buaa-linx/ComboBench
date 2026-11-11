# Icon sources and notices

Downloaded on 2026-10-30. These are local SVG assets, so the webpage does not depend on an external icon service at runtime.

| File | Source | Notice |
| --- | --- | --- |
| `paper.svg` | [Lucide file-text](https://lucide.dev/icons/file-text), [upstream SVG](https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/file-text.svg) | Lucide license; see `LUCIDE-LICENSE.txt`. |
| `data.svg` | [Lucide database](https://lucide.dev/icons/database), [upstream SVG](https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/database.svg) | Lucide / Feather-derived licensing; complete upstream notices are preserved in `LUCIDE-LICENSE.txt`. |
| `video.svg` | [Lucide circle-play](https://lucide.dev/icons/circle-play), [upstream SVG](https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/circle-play.svg) | Lucide license; see `LUCIDE-LICENSE.txt`. |
| `github.svg` | [GitHub official Brand Toolkit](https://brand.github.com/foundations/logo), [official download archive](https://brand.github.com/GitHub_Logos.zip), archive entry `GitHub Logos/SVG/GitHub_Invertocat_White.svg` | Official white Invertocat, preserved byte-for-byte. GitHub trademarks remain GitHub, Inc.'s property. |
| `github-black.svg` | Same official archive, entry `GitHub Logos/SVG/GitHub_Invertocat_Black.svg` | Official black variant, preserved byte-for-byte; optional alternative for light backgrounds. |
| `arxiv.svg` | [Simple Icons arXiv source](https://github.com/simple-icons/simple-icons/blob/develop/icons/arxiv.svg), [upstream SVG](https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/arxiv.svg) | Library distributed under CC0; see `SIMPLE-ICONS-LICENSE.md`. The arXiv mark remains its owner's trademark. Reserved for future use, not a fourth hero button. |

## Integration

- The hero should show exactly **Paper / Code / Data**. Code uses the GitHub white Invertocat; the arXiv file is kept in reserve.
- Lucide icons use `stroke="currentColor"` and a 24×24 viewBox. Inline SVG inherits the surrounding text color. If used as an external `<img>` on a dark button, apply `filter: brightness(0) invert(1)` to these functional icons.
- `github.svg` is already the official white artwork. Give it an approximately 19×19 CSS box with `object-fit: contain`; do not distort its 98×96 proportions or apply the Lucide inversion filter.
- GitHub's official guidance permits its logo as a link to a GitHub profile or project, calls for secondary placement, and specifies high contrast without graphic effects. It must not be used as the ComboBench project logo or to imply endorsement.
- Two video placeholders can share the same Lucide `video.svg` icon. Keep the dataset names (nuScenes / KITTI) as text rather than inventing dataset brand marks.
