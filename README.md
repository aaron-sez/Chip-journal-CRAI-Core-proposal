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

================================EXECUTION================================
## 🚀 Getting Started and Execution Steps

To set up the simulation environment and execute the CRAI-Core framework locally, follow the steps below:

### 1. Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone [https://github.com/your-username/CRAI-Core.git](https://github.com/your-username/CRAI-Core.git)
cd CRAI-Core

2. Create a Virtual Environment (Recommended, to work with Python and to install dependencies cleanly, etc.)
# On macOS and Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
Install the required Python packages specified in the manifest:
pip install --upgrade pip
pip install -r requirements.txt

4. Run the Master Simulation Pipeline
To execute the complete 13-node CRAI-Core simulation pipeline (covering Phase 1 profiling, Phase 2 compilation, and Phase 3 runtime execution):
python crai_core_simulator.py --model "ResNet-50" --accuracy 76.1 --seu_rate 1e-3

5. Run the SEU Fault-Resilience Stress Test
To reproduce the Monte Carlo single-event upset (SEU) stress-testing evaluation across custom error rates:
python experiments/run_seu_stress.py --error_rate 1e-3 --runs 5000
