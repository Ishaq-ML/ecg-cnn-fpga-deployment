module test_all_samples_rom #(
    parameter INIT_FILE  = "ise/combined_test_inputs.hex",
    parameter SAMPLE_LEN = 180,
    parameter N_SAMPLES  = 30
)(
    input  wire        clk,
    input  wire        reset,
    input  wire [4:0]  sw,
    output reg         data_valid_out,
    output reg  signed [15:0] data_out,
    output reg         done
);
    localparam TOTAL = N_SAMPLES * SAMPLE_LEN;

    (* RAM_STYLE = "BLOCK" *)
    reg [15:0] mem [0:TOTAL-1];
    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem, 0, TOTAL-1);
    end

    reg [12:0] rd_addr;
    reg [15:0] rd_data;
    always @(posedge clk)
        rd_data <= mem[rd_addr];

    reg [12:0] base_addr;
    reg [7:0]  cnt;
    reg        running;
    reg [4:0]  sw_prev;

    always @(posedge clk) begin
        if (reset) begin
            base_addr <= 0; cnt <= 0; running <= 0;
            rd_addr <= 0; data_valid_out <= 0; data_out <= 0;
            done <= 0; sw_prev <= sw;
        end else begin
            done <= 0;
            base_addr <= (sw < N_SAMPLES) ? (sw * SAMPLE_LEN) : 0;

            if (sw != sw_prev || (!running && cnt == 0)) begin
                sw_prev <= sw; cnt <= 0; running <= 1;
                rd_addr <= base_addr; data_valid_out <= 0;
            end else if (running) begin
                if (cnt < SAMPLE_LEN) begin
                    rd_addr <= base_addr + cnt;
                    if (cnt > 0) begin data_valid_out <= 1; data_out <= rd_data; end
                    cnt <= cnt + 1;
                end else begin
                    data_valid_out <= 1; data_out <= rd_data;
                    done <= 1; running <= 0; cnt <= 0; data_valid_out <= 0;
                end
            end else begin
                data_valid_out <= 0;
            end
        end
    end
endmodule
