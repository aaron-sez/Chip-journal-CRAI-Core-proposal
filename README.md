CRAI-Core/
│
├── README.md               # Project documentation and setup instructions
├── requirements.txt        # Python dependency manifest (NumPy, PyTorch, etc.)
├── LICENSE                 # Open Source License
├── crai_core_simulator.py  # Master orchestration framework bridging all 13 nodes
│
├── profiling/              # Phase 1: Profiling and Optimization
│   └── sensitivity.py      # Layer-wise gradient perturbation profiler (Node 2)
│
├── compiler/               # Phase 2: Compilation and Mapping
│   └── allocator.py        # Scratchpad memory and reuse buffer allocator (Node 8)
│
└── experiments/            # Benchmark execution and evaluation scripts
    └── run_seu_stress.py   # Monte Carlo SEU fault resilience stress testing

    =======================================EXPLANATION=======================================

Step-by-Step Repository Overview
Here is a clean file breakdown from start to finish:
1) requirements.txt: Lists all necessary dependencies (numpy, torch, matplotlib, scipy) required to execute the simulation and experiment scripts. Note: Based on the date one needs to execute this framework code, updates may be required.
2) crai_core_simulator.py (Root Directory): The master orchestration script shown in the root folder connects all 13 functional nodes across the three sequential phases.
3) profiling/sensitivity.py (Node 2): Specialized module dedicated to gradient perturbation and layer-wise sensitivity evaluation.
4) compiler/allocator.py (Node 8): Dedicated module managing scratchpad SRAM layouts and reuse buffers to minimize off-chip DRAM bottlenecks.
5) experiments/run_seu_stress.py: A dedicated Monte Carlo experiment runner that executes stress tests across SEU error rates ($10^{-6}$ to $10^{-3}$) to generate the resilience curves detailed in our paper.
