"""
Filename: compiler/allocator.py
Description: Optimizes scratchpad memory layout and data reuse buffers (Node 8).
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MemoryAllocator:
    def __init__(self, total_scratchpad_kb: int = 512):
        self.total_scratchpad_kb = total_scratchpad_kb
        self.allocated_buffers: Dict[str, int] = {}

    def optimize_buffer_allocation(self, layer_precisions: Dict[str, str]) -> Dict[str, int]:
        """
        Allocates SRAM scratchpad space based on layer precision footprints (FP16, INT8, INT4).
        """
        logging.info("Allocating scratchpad memory (Limit: %d KB)...", self.total_scratchpad_kb)
        
        byte_sizes = {"FP16": 2, "INT8": 1, "INT4": 0.5}
        current_usage = 0
        
        for layer, precision in layer_precisions.items():
            # Estimate buffer size based on typical feature map dimensions
            buffer_size_kb = int(64 * byte_sizes.get(precision, 1))
            if current_usage + buffer_size_kb <= self.total_scratchpad_kb:
                self.allocated_buffers[layer] = buffer_size_kb
                current_usage += buffer_size_kb
            else:
                self.allocated_buffers[layer] = 0 # Offload to external memory
                logging.warning("Scratchpad full! Layer %s offloaded to external DRAM.", layer)
                
        logging.info("Memory allocation complete. Total scratchpad utilized: %d KB", current_usage)
        return self.allocated_buffers
