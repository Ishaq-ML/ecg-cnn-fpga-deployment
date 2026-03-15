import os
import csv

def main():
    base_dir = "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/My Drive/NHSNN Personal Drive/FPGA/projects/ECG/pareto_03FPGA"
    src_dir = os.path.join(base_dir, "ECG_optimal_outputs")
    manifest_csv = os.path.join(src_dir, "test_manifest.csv")
    
    rtl_hex_dir = os.path.join(base_dir, "RTL", "hex")
    out_test_dir = os.path.join(rtl_hex_dir, "test_inputs_1")
    os.makedirs(out_test_dir, exist_ok=True)
    
    out_manifest = os.path.join(out_test_dir, "test_manifest.txt")
    
    with open(manifest_csv, "r") as f:
        reader = csv.DictReader(f)
        with open(out_manifest, "w") as mf:
            mf.write("# coe_file, expected_class_id\n")
            for row in reader:
                fname = row['file_name']
                cid = row['ground_truth_id']
                mf.write(f"{fname},{cid}\n")
                
                # Convert coe to hex
                coe_path = os.path.join(src_dir, fname)
                hex_path = os.path.join(out_test_dir, fname.replace('.coe', '.hex'))
                
                if os.path.exists(coe_path):
                    with open(coe_path, 'r') as cf:
                        lines = cf.readlines()
                    
                    data_start = False
                    hex_values = []
                    for line in lines:
                        if "memory_initialization_vector" in line:
                            data_start = True
                            continue
                        if data_start:
                            clean_line = line.replace(',', '').replace(';', '').strip()
                            if clean_line:
                                int_val = int(clean_line)
                                # val_q88 = round((val_int - ZP) * Scale * 256)
                                # model: Scale=0.03813249, ZP=-25.  Scale*256 approx 9.76
                                val_float = (int_val - (-25)) * 0.03813249
                                val_q88 = int(round(val_float * 256.0))
                                
                                if val_q88 < 0:
                                    val_q88 = (1 << 16) + val_q88
                                else:
                                    val_q88 = val_q88 & 0xFFFF
                                hex_values.append(f"{val_q88:04x}")
                                
                    with open(hex_path, 'w') as hf:
                        hf.write("\n".join(hex_values))
                        
    print("Test samples extraction and conversion complete.")

if __name__ == "__main__":
    main()
