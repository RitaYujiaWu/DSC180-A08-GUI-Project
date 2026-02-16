#!/usr/bin/env python3
"""
Script to move files listed in successful_path.txt to organized target folders
based on their domain (academic, finance, shopping, etc.)

Usage: python move_successful_files.py
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('move_files.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

path = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/training_data/expand_memory'
all_subsets = os.listdir(path)
all_subsets = [subset for subset in all_subsets if os.path.isdir(os.path.join(path, subset))]
all_subsets = [subset.split('_')[0] for subset in all_subsets]
all_subsets = list(set(all_subsets))
print(all_subsets)
    
def extract_domain_from_path(file_path: str) -> str:
    """
    Extract domain from file path.
    Expected format: /home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/training_data/expand_memory/{domain}/...
    
    Args:
        file_path: Full path to the file
        
    Returns:
        Domain name (e.g., 'finance', 'academic', 'shopping')
    """
    for subset in all_subsets:
        if subset in file_path:
            return subset
    print(f"Could not extract domain from path: {file_path}")
    return 'unknown'

def create_target_directory(domain: str, base_target: str) -> str:
    """
    Create target directory for the domain if it doesn't exist.
    
    Args:
        domain: Domain name
        base_target: Base target directory path
        
    Returns:
        Full path to the target directory
    """
    target_dir = os.path.join(base_target, domain, "success")
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Created/verified target directory: {target_dir}")
        return target_dir
    except Exception as e:
        logger.error(f"Failed to create target directory {target_dir}: {e}")
        raise

def move_file(source_path: str, target_dir: str) -> bool:
    """
    Move a file from source to target directory.
    
    Args:
        source_path: Full path to source file
        target_dir: Target directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if source file exists
        if not os.path.exists(source_path):
            logger.warning(f"Source file does not exist: {source_path}")
            return False
        
        # Get filename from source path
        filename = os.path.basename(source_path)
        new_filename = filename.split('.')[0] + '_0.jsonl'
        target_path = os.path.join(target_dir, new_filename)
        
        # Check if target file already exists
        if os.path.exists(target_path):
            logger.warning(f"Target file already exists: {target_path}")
            new_filename = filename.split('_')[0] + '_1.jsonl'
            target_path = os.path.join(target_dir, new_filename)
        
        # Move the file
        shutil.copy(source_path, target_path)
        logger.info(f"Moved: {source_path} -> {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to move file {source_path}: {e}")
        return False

def read_successful_paths(file_path: str) -> List[str]:
    """
    Read file paths from the successful_path.txt file.
    
    Args:
        file_path: Path to the successful_path.txt file
        
    Returns:
        List of file paths
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(paths)} file paths from {file_path}")
        return paths
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise

def main():
    """Main function to orchestrate the file moving process."""
    
    # Configuration
    successful_paths_file = "/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/count_result/successful_path.txt"
    base_target_dir = "/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/expand_memory_organized/expand_memory"
    
    # Statistics
    stats = {
        'total_files': 0,
        'moved_successfully': 0,
        'failed_to_move': 0,
        'files_not_found': 0,
        'target_exists': 0,
        'domain_stats': {}
    }
    
    try:
        # Read file paths
        logger.info("Starting file moving process...")
        file_paths = read_successful_paths(successful_paths_file)
        stats['total_files'] = len(file_paths)
        
        # Process each file
        for i, file_path in enumerate(file_paths, 1):
            if i % 100 == 0:
                logger.info(f"Processing file {i}/{len(file_paths)}")
            
            # Extract domain
            domain = extract_domain_from_path(file_path)
            
            # Update domain statistics
            if domain not in stats['domain_stats']:
                stats['domain_stats'][domain] = 0
            
            # Create target directory
            try:
                target_dir = create_target_directory(domain, base_target_dir)
            except Exception as e:
                logger.error(f"Failed to create target directory for domain {domain}: {e}")
                stats['failed_to_move'] += 1
                continue
            
            # Move file
            if move_file(file_path, target_dir):
                stats['moved_successfully'] += 1
                stats['domain_stats'][domain] += 1
            else:
                # Check why it failed
                if not os.path.exists(file_path):
                    stats['files_not_found'] += 1
                else:
                    # Check if target already exists
                    filename = os.path.basename(file_path)
                    target_path = os.path.join(target_dir, filename)
                    if os.path.exists(target_path):
                        stats['target_exists'] += 1
                    else:
                        stats['failed_to_move'] += 1
            # if i > 100:
            #     break
        
        # Print final statistics
        logger.info("=" * 50)
        logger.info("MOVING PROCESS COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {stats['total_files']}")
        logger.info(f"Successfully moved: {stats['moved_successfully']}")
        logger.info(f"Files not found: {stats['files_not_found']}")
        logger.info(f"Target files already exist: {stats['target_exists']}")
        logger.info(f"Failed to move: {stats['failed_to_move']}")
        logger.info("")
        logger.info("Files moved by domain:")
        for domain, count in sorted(stats['domain_stats'].items()):
            logger.info(f"  {domain}: {count} files")
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
