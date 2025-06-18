# Local Neural Operators for Equation-Free System-level analysis

This repository contains code for the paper:

**"Enabling Local Neural Operators to perform Equation-Free System-Level Analysis"**
*G. Fabiani, H. Vandecasteele, S. Goswami, C. Siettos, I.G. Kevrekidis*
\[[arXiv:2505.02308](https://arxiv.org/abs/2505.02308)]

We integrate **local Neural Operators (NOs)** with iterative numerical methods in the **Krylov subspace** to enable fixed-point computation, stability, and bifurcation analysis in nonlinear PDEs - without requiring closed-form equations.
We apply the framework to:

* 1D Allen-Cahn equation
* Liouville-Bratu-Gelfand equation
* FitzHugh-Nagumo system (simulated through a Lattice-Boltzmann method (LBM), thus learning the NO from "mesoscopic" data)

If you use or adapt this code for your research, please cite our paper.

---

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
Licensed under **Creative Commons BY-NC-SA 4.0**.
Commercial use is not permitted.

---

## Disclaimer

This software is provided *"as is"* without warranties. The authors are not liable for any damages resulting from its use.

---

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
