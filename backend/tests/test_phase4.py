import os
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from main import solve_sudoku_image

def main():
    images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sudoku_images_phase4'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/output/phase4'))
    final_test_dir = os.path.join(output_dir, 'final_test')
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tmp'))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/digit_recognizer.pth'))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_test_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, 'test_output.txt')
    
    with open(log_file_path, 'w') as log_file:
        images = sorted(glob.glob(os.path.join(images_dir, "*.*")))
        if not images:
            msg = "No images found in data/sudoku_images_phase4!"
            print(msg)
            log_file.write(msg + "\n")
            return
            
        for img_path in images:
            basename = os.path.basename(img_path)
            out_path = os.path.join(final_test_dir, f"final_{basename}")
            
            # Redirect stdout briefly to capture prints from solve_sudoku_image
            import io
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            result = solve_sudoku_image(img_path, model_path, out_path, None)
            success = result[0] if isinstance(result, tuple) else False
            
            sys.stdout = sys.__stdout__
            output_str = captured_output.getvalue()
            
            print(output_str, end='')
            log_file.write(output_str)
            
            msg3 = "========================================\n\n"
            print(msg3, end='')
            log_file.write(msg3)

if __name__ == "__main__":
    main()
