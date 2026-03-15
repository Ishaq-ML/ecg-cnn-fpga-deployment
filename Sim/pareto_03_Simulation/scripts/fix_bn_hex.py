import os

def main():
    hex_dir = "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/My Drive/NHSNN Personal Drive/FPGA/projects/ECG/pareto_03FPGA/RTL/hex"
    
    # We create dummy BN files with exactly 64 lines to satisfy layer_batchnorm.v [0:63]
    for i in range(1, 5):
        w_file = os.path.join(hex_dir, f"conv{i}_bn_w.hex")
        b_file = os.path.join(hex_dir, f"conv{i}_bn_b.hex")
        
        with open(w_file, "w") as f:
            f.write("\n".join(["0100" for _ in range(64)]))
            
        with open(b_file, "w") as f:
            f.write("\n".join(["0000" for _ in range(64)]))
            
    print("Fixed dummy BN hex files to have 64 elements each.")

if __name__ == "__main__":
    main()
