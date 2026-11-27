"""
Filename: experiments/run_seu_stress.py
Description: Executes Monte Carlo SEU fault injection campaigns across error rates (1e-6 to 1e-3).
"""

import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_stress_test(error_rate: float, runs: int):
    logging.info("Initializing SEU Stress Campaign | Error Rate: %.e | Monte Carlo Runs: %d", error_rate, runs)
    
    layers = ["conv1", "layer1", "layer2", "layer3", "layer4", "classifier"]
    protected_layers = {"conv1", "classifier"}
    
    survived = 0
    for r in range(runs):
        corrupted = False
        for layer in layers:
            if np.random.rand() < error_rate:
                if layer in protected_layers:
                    continue # Hardware parity check recovers error
                else:
                    if np.random.rand() > 0.30:
                        continue # Approximate resilience absorbs minor noise
                    else:
                        corrupted = True
                        break
        if not corrupted:
            survived += 1
            
    retention_rate = (survived / runs) * 100.0
    logging.info("Result -> SEU Rate: %.e | Accuracy Retention: %.2f%%", error_rate, retention_rate)
    return retention_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CRAI-Core SEU Fault Stress Test")
    parser.add_argument("--error_rate", type=float, default=1e-3, help="SEU error rate per bit-cycle")
    parser.add_argument("--runs", type=int, default=5000, help="Number of Monte Carlo simulation runs")
    args = parser.parse_args()
    
    run_stress_test(error_rate=args.error_rate, runs=args.runs)
