# GEN3D report (LaTeX)

FYP report kept up to date as work progresses.

## Compile
**Overleaf (easiest):** zip the whole `docs/report` folder → Overleaf → New Project → Upload Project → Recompile.

**Locally** (needs MiKTeX or TeX Live):
```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Files
| File | Content |
|---|---|
| `main.tex` | title page, packages, chapter order. **Fill in names/supervisor at the top.** |
| `chapters/00_abstract.tex` … `08_conclusion.tex` | one file per chapter |
| `references.bib` | IEEE references (rubric: at least 25, all cited) |
| `figures/` | images used in the report (copied from `outputs/`) |

## How to update
- Search for `\todo{` to find every open item (shown in red in the PDF).
- New result → add numbers to `05_results.tex`, image to `figures/`.
- New problem solved → add a row to `06_issues.tex` (and `docs/technical_issues_log.md`).
- Meetings / work split / Gantt → `07_management.tex`.
