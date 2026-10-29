# Scripts from the 2026-10-19 thesis session

Scripts that were run in the session to make the CFD, imbalance, convergence, air-volume and hardware figures of the thesis. They were copied from the session transcript (`01`–`34`) and from the session scratch folder (`thread_*`). The originals were not changed.

They are not standalone: most of them write to hard-coded paths in the thesis folder (`Thesis_Latex/Git/...`, `ANSYS Figures/`) or to the current folder. Check the paths before running them again.

| Script | What it does | Output |
|---|---|---|
| `01`–`02` | Composes CFD image pairs with a shared colour bar; `02` runs the composition for the other pairs | CFD pair figures (`figures/07_results`) |
| `03`, `05` | Velocity pair and velocity in the silencer | `cfd_velocity_pair.png` |
| `04`, `06`–`09` | Pressure midplane pairs with equal disk sizes | `cfd_midplane_pair.png` |
| `10`–`11` | Y+ figure with a separate legend bar | `cfd_yplus_pair.png` |
| `12`–`13` | Axis triad image used to clean the CFD screenshots | `triad.png` (intermediate) |
| `14` | Extracts the net mass imbalance of all runs from the report files | `imbalance.json` |
| `15`–`20`, `23`–`25` | Net mass imbalance plots (versions, final version, UID08 and UID12 cases) | `imbalance_*.pdf` |
| `21` | UID12 update of the imbalance data | `imbalance_uid12_updated.pdf` |
| `22` | Reads the report zip and parses the metrics | `new_metrics.json` |
| `26` | Plots the convergence residual files of one case (`uid19.py` in the thread folder is the same kind of script) | convergence figure |
| `27`–`29`, `31` | Air-volume pair from the CAD model (native scale, block tops aligned) | `cfd_air_volume_pair.png` |
| `30` | Cleans and crops the labelled domain figure | `cfd_domain_labelled.png` |
| `33`–`34` | Labels and resizes the WFE hardware and controller figures | `wfe_hardware.png`, `wfe_controller.png` |
| `thread_compose_pairs.py` | Pair composition used by `01`–`02` | — |
| `thread_legend_middle.py`, `thread_swap_circles.py` | Legend and circle placement for the 07_results figures | — |
| `thread_render_slides.py` | Renders selected pages of a PDF (progress slides) to PNG | — |
| `thread_uid19.py` | Plots the residual files of one case | — |
| `thread_reorg.py` | Moves the figures into the chapter folders and rewrites their paths in the .tex files | changes the thesis folder; check before running |

Not included: one-line checks and LaTeX text edits made in the same session.
