# Local Neural Operators for Equation-Free System-level analysis

This repository contains code for the paper:

**"Enabling Local Neural Operators to perform Equation-Free System-Level Analysis"**
*G. Fabiani, H. Vandecasteele, S. Goswami, C. Siettos, I.G. Kevrekidis*
\[[arXiv:2505.02308](https://arxiv.org/abs/2505.02308)]

We integrate **local Neural Operators (NOs)** with iterative numerical methods in the **Krylov subspace** to enable fixed-point computation, stability, and bifurcation analysis in nonlinear PDEs - without requiring closed-form equations.
We apply the framework to:

* 1D Allen-Cahn equation
* Liouville-Bratu-Gelfand equation (see Folder Parabolic Bratu-Gelfand)
* FitzHugh-Nagumo system (simulated through a Lattice-Boltzmann method (LBM), thus learning the NO from "mesoscopic" data)
(see Folder with LBM)

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

---

## Abstract
Neural Operators (NOs) provide a powerful framework for computations involving physical laws that can be modelled by (integro-) partial differential equations (PDEs), directly learning maps between infinite-dimensional function spaces that bypass both the explicit equation identification and their subsequent numerical solving. Still, NOs have so far primarily been employed to explore the dynamical behavior as surrogates of brute-force temporal simulations/predictions. Their potential for systematic rigorous numerical system-level tasks, such as fixed-point, stability, and bifurcation analysis - crucial for predicting irreversible transitions in real-world phenomena - remains largely unexplored. Toward this aim, inspired by the Equation-Free multiscale framework, we propose and implement a framework that integrates (local) NOs with advanced iterative numerical methods in the Krylov subspace, so as to perform efficient system-level stability and bifurcation analysis of large-scale dynamical systems. Beyond fixed point, stability, and bifurcation analysis enabled by local in time NOs, we also demonstrate the usefulness of local in space as well as in space-time ("patch") NOs in accelerating the computer-aided analysis of spatiotemporal dynamics. We illustrate our framework via three nonlinear PDE benchmarks: the 1D Allen-Cahn equation, which undergoes multiple concatenated pitchfork bifurcations; the Liouville-Bratu-Gelfand PDE, which features a saddle-node tipping point; and the FitzHugh-Nagumo (FHN) model, consisting of two coupled PDEs that exhibit both Hopf and saddle-node bifurcations.
