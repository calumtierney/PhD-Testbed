# Proposed CLAUDE.md §4f (paste after §4e, before "## 5. Build order")

The task prompt asked for this subsection to be added to CLAUDE.md. The edit was blocked in the session that wrote it, and CLAUDE.md is yours, so it is here instead. §4a–§4e are untouched. Every value below was checked against the paper's PDF (Hughes et al. 2011, Meas. Sci. Technol. 22 045103, Tables 2–4 and the text of sections 5–8).

---

### 4f. The NPL geometric model — Hughes et al. 2011 (Meas. Sci. Technol. 22 045103), Tables 2–4

Implemented in `instruments/npl_model.py`; see `docs/step1b_npl_geometric_model.md`.

**Table 2, distances and scale (k = 1):**

| Symbol | Value | Uncertainty |
|---|---|---|
| λ | −2.4676 mm | 0.403 µm |
| μ | printed "1.000" | 5 × 10⁻⁷ |
| e_x | 4.988 µm | 0.122 µm |
| b_y,0 | 91.201 µm | 0.654 µm |
| b_z,0 | −13.318 µm | 0.974 µm |

The observation equation is d = (1 + μ)·d* + λ, so the printed 1.000 can only mean 1 + μ = 1.000, i.e. μ ≈ 0. `TODO(source)`: the paper gives no more digits.

**Table 3, angles in arcseconds (k = 1):**

| Symbol | Value | Uncertainty |
|---|---|---|
| α | 16.751 | 0.128 |
| γ | 9.647 | 0.079 |
| a_A,1 | 0.526 | 0.064 |
| b_A,1 | −0.562 | 0.080 |
| a_A,2 | 0.177 | 0.073 |
| b_A,2 | 1.530 | 0.090 |
| a_E,0 | 0.154 | 0.223 |
| a_E,1 | −1.736 | 0.152 |
| b_E,1 | −0.677 | 0.183 |
| a_E,2 | 0.375 | 0.214 |
| b_E,2 | 0.437 | 0.179 |

These describe an API T3 **with its internal compensation switched off**, measured in IFM mode: the errors the network test found, not the errors that remain after compensation. The paper reports the correlations between parameters only in words: the significant ones are γ–b_y, a_E,0–b_z and a_E,2–a_E,0 (the last because no elevation in the test exceeded 57°, mean cos 2φ = 0.69). No magnitudes are given and none may be invented. Constraints: a_A,0 = 0 fixes the frame; b_x,0 = 0 (inseparable from λ); with n_Q = 2 there are 14 geometric parameters, plus λ and μ.

**Other reference results from the same paper**, for gates:
- Network: 15 targets, 5 tracker positions, 126 readings out of a possible 15 × 5 × 2 = 150 (so about 63 target-position pairs, each read on both faces). Ranges 0.27–6.4 m; elevations no higher than 57° (mechanical range −60° to +77°); 55 min.
- Layout: 8 targets on two walls meeting at a corner, 1 on the floor, 6 on a bench. Tracker 0 sets the frame; 1 is at the same place as 0 but rotated; 2 is on the stand at full height; 3 on the bench; 4 on a floor trivet. Two "birdbath" lines break the λ–e_x correlation: targets 6 & 8 (viewed from positions 3 and 0) and 9 & 10 (positions 0 and 4).
- Fitted target uncertainties 2.1–4.9 µm, tracker positions 1.1–2.2 µm, pointing directions 0.33–0.45″.
- A simulation of 1000 points at 1–5 m, random azimuth and elevation, front and back face: about 80% of the variance from the angle sensors' random effects, about 20% from the parameters h (0.1–3.8% each).
- The model applies to trackers with the laser source in the rotating head (current Leica, FARO, API). It does not apply to older gimballed-mirror designs.
- Table 4 labels each sensor's noise "A" and "R" (σ_D,A, σ_D,R, …); the 0.312″ distance row of §4a is still unexplained. Open question for NPL.
