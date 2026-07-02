import os
import sys
import glob

# Add src to the Python path so we can import extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from extractor import extract_sudoku

def main():
    # Updated paths to be absolute based on the script location
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sudoku_images'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/output/phase1'))
    
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found!")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    samples = sorted(glob.glob(os.path.join(samples_dir, "*.*")))
    if not samples:
        print("No samples found!")
        return
        
    for sample in samples:
        basename = os.path.basename(sample)
        name, _ = os.path.splitext(basename)
        
        print(f"Processing {basename}...")
        
        sample_output_dir = os.path.join(output_dir, name)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        cells = extract_sudoku(sample, sample_output_dir)
        if cells is not None:
            print(f"Successfully processed {basename}!\n")
        else:
            print(f"Failed to process {basename}!\n")

if __name__ == "__main__":
    main()
