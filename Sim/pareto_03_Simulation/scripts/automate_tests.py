import os
import subprocess
import shutil
import re

def run_test_suite(test_dir_name, rtl_dir, results_file):
    test_dir = os.path.join(rtl_dir, "hex", test_dir_name)
    manifest_path = os.path.join(test_dir, "test_manifest.txt")
    target_hex = os.path.join(rtl_dir, "hex", "clean_input.hex")
    sim_cmd = ["vvp", "simv"]
    
    if not os.path.exists(manifest_path):
        results_file.write(f"Manifest not found in {test_dir}\n")
        return

    results_file.write(f"\n--- Running Tests for {test_dir_name} ---\n")
    results_file.write(f"{'Sample':<25} | {'Expected':<10} | {'Actual':<10} | {'Status':<10}\n")
    results_file.write("-" * 65 + "\n")

    correct = 0
    total = 0

    print(f"Compiling Verilog files in {rtl_dir}...")
    v_files = ["tb/tb_pareto03.v", "top_pareto03.v"]
    for d in ["block1_conv8", "block2_conv32", "block3_conv16", "block4_conv16", "block5_dense", "common"]:
        d_path = os.path.join(rtl_dir, d)
        if os.path.exists(d_path):
            for f in os.listdir(d_path):
                if f.endswith(".v"):
                    v_files.append(f"{d}/{f}")

    res = subprocess.run(["iverilog", "-o", "simv"] + v_files, cwd=rtl_dir, capture_output=True, text=True, timeout=120.0, stdin=subprocess.DEVNULL)
    if res.returncode != 0:
        print("Compilation failed!")
        print(res.stderr)
        return

    print("Compilation complete. Running simulations...")

    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
                
            coe_file = parts[0]
            expected_id = parts[1]
            hex_file = coe_file.replace('.coe', '.hex')
            hex_path = os.path.join(test_dir, hex_file)
            
            if not os.path.exists(hex_path):
                results_file.write(f"{hex_file:<25} | {expected_id:<10} | {'MISSING':<10} | FAIL\n")
                continue

            shutil.copy(hex_path, target_hex)
            
            try:
                process = subprocess.run(sim_cmd, cwd=rtl_dir, capture_output=True, text=True, timeout=15.0, stdin=subprocess.DEVNULL)
                stdout = process.stdout
                
                match = re.search(r"Classification Result: Class\s+(\d+)", stdout)
                if match:
                    actual_id = match.group(1)
                else:
                    actual_id = "TIMEOUT_V" if "TIMEOUT" in stdout else "ERROR"
            except subprocess.TimeoutExpired:
                actual_id = "HANG"
            except Exception as e:
                actual_id = f"EXCEPTION"
            
            status = "PASS" if actual_id == expected_id else "FAIL"
            if status == "PASS":
                correct += 1
            total += 1
            
            print(f"{hex_file:<25} | {expected_id:<10} | {actual_id:<10} | {status:<10}")
            results_file.write(f"{hex_file:<25} | {expected_id:<10} | {actual_id:<10} | {status:<10}\n")
            results_file.flush()

    accuracy = (correct / total) * 100 if total > 0 else 0
    error_rate = 100 - accuracy
    results_file.write("-" * 65 + "\n")
    results_file.write(f"Summary for {test_dir_name} (4-Class pareto_03):\n")
    results_file.write(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%, Error Rate: {error_rate:.2f}%\n")

def main():
    rtl_dir = "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/My Drive/NHSNN Personal Drive/FPGA/projects/ECG/pareto_03FPGA/RTL"
    output_path = os.path.join(rtl_dir, "test_results.txt")
    
    with open(output_path, 'w') as results_file:
        results_file.write("ECG Classifier (pareto_03) Results\n")
        results_file.write("=" * 70 + "\n")
        run_test_suite("test_inputs_1", rtl_dir, results_file)
        
    print(f"\nTest suite completed. Results written to {output_path}")

if __name__ == "__main__":
    main()
