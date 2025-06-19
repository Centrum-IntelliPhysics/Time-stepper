# Local Neural Operators for Equation-Free System-level analysis

This repository contains code for the paper:

**"Enabling Local Neural Operators to perform Equation-Free System-Level Analysis"**
*G. Fabiani, H. Vandecasteele, S. Goswami, C. Siettos, I.G. Kevrekidis*
\[[arXiv:2505.02308](https://arxiv.org/abs/2505.02308)]

If you use or adapt this code for your research, please cite our paper.

⭐ An alternative title for this project could have been:
“Every numerical task you wished your Neural Operator could perform, and you did not know it could”.

This software implements data-driven **local Neural Operators (NOs)** trained on **short-time data**, learning **local time-stepping dynamics** (local in time). It also supports **local-in-space learning** via the **gap-tooth scheme**: a domain partitioning method with inactive gaps between patches.

We use state-of-the-art NOs, including **RandONets**, and compare them with **DeepONets**, for efficient operator learning from data.

We integrate **local Neural Operators (NOs)** with iterative numerical methods in the **Krylov subspace** to enable fixed-point computation, stability, and bifurcation analysis in nonlinear PDEs - without requiring closed-form equations. All bypassing derivation of explicit equations and their numerical solution. 🔥


<p align="center">
  <img src="https://raw.githubusercontent.com/Centrum-IntelliPhysics/local-neural-operator-time-stepper-instead-of-just-time-stepper/refs/heads/main/images/Schematic_Overview_4.png" width="600"/>
</p>

We apply the framework to:

* 1D Allen-Cahn equation
* Liouville-Bratu-Gelfand equation (see Folder Parabolic Bratu-Gelfand)
* FitzHugh-Nagumo system (simulated through a Lattice-Boltzmann method (LBM), thus learning the NO from "mesoscopic" data)
(see Folder with LBM)

The repository also includes some additional results:
* FHN simulated with centered Finite Differences, with training using DeepONets.
* Allen-Cahn tested with both DeepONets and Fourier Neural Operators.
* Some preliminary DeepONet results for the Bratu problem.
* Some attempt on a simple diffusion equation.

These results are not part of the paper, and the code is not well-documented and may not run properly.
They were part of early tests showing that high accuracy is critical for system-level tasks like computing steady-state eigenvalues — which is why they were eventually excluded.

Highlights
====
⚡Beyond simple simulations: Efficient bifurcation tracking and stability analysis from data 

🧠Matrix-free methods in the Krylov subspace for steady-state (Newton-GMRES) stability (Arnoldi) and bifurcation tracking (arc-length continuation) 

🌐A powerful framework for high-dimensional, complex systems without relying on traditional PDEs or explicit models

🔄Introduction of a Homotopy-based Embedding for handling parameter-dependent operator problems 

🏆In our illustrations, we used RandONets, a randomized neural operator, trained, in all examples, over large datasets in less than a minute (!), obtaining surprising accuracy! 

(We also compared with DeepONets, which benefit from our framework, but exhibited limited accuracy in stability analysis in our experiments. 🧐)

Abstract of the paper
=====
Neural Operators (NOs) provide a powerful framework for computations involving physical laws that can be modelled by (integro-) partial differential equations (PDEs), directly learning maps between infinite-dimensional function spaces that bypass both the explicit equation identification and their subsequent numerical solving. Still, NOs have so far primarily been employed to explore the dynamical behavior as surrogates of brute-force temporal simulations/predictions. Their potential for systematic rigorous numerical system-level tasks, such as fixed-point, stability, and bifurcation analysis - crucial for predicting irreversible transitions in real-world phenomena - remains largely unexplored. Toward this aim, inspired by the Equation-Free multiscale framework, we propose and implement a framework that integrates (local) NOs with advanced iterative numerical methods in the Krylov subspace, so as to perform efficient system-level stability and bifurcation analysis of large-scale dynamical systems. Beyond fixed point, stability, and bifurcation analysis enabled by local in time NOs, we also demonstrate the usefulness of local in space as well as in space-time ("patch") NOs in accelerating the computer-aided analysis of spatiotemporal dynamics. We illustrate our framework via three nonlinear PDE benchmarks: the 1D Allen-Cahn equation, which undergoes multiple concatenated pitchfork bifurcations; the Liouville-Bratu-Gelfand PDE, which features a saddle-node tipping point; and the FitzHugh-Nagumo (FHN) model, consisting of two coupled PDEs that exhibit both Hopf and saddle-node bifurcations.

---

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
You may not use this material for commercial purposes.
If you remix, transform, or build upon this material, you must distribute your contributions under the same license as the original.
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---

## Disclaimer

This software is provided "as is" without any express or implied warranties.
This includes, but is not limited to, warranties of merchantability, fitness for a particular purpose, and non-infringement.
The authors and copyright holders are not liable for any claims, damages, or other liabilities arising from the use of this software

Copyright (c) 2025 - The Authors

Last revised by G. Fabiani, June 19, 2025

---

For details on RandONets, see:
👉 [https://github.com/GianlucaFabiani/RandONets](https://github.com/GianlucaFabiani/RandONets)
