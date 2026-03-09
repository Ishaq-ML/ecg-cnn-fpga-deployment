module layer7_dense #(
    parameter WEIGHT_FILE = "hex/w_dense.hex",
    parameter BIAS_FILE   = "hex/b_dense.hex"
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [1023:0] data_in,
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] weights [0:63];
    reg signed [15:0] biases  [0:0];

    // Combinational MAC
    reg signed [31:0] sum;
    integer i;
    always @(*) begin
        sum = (biases[0] <<< 8);
        for (i = 0; i < 64; i = i + 1)
            sum = sum + ($signed(data_in[i*16 +: 16]) * weights[i]);
    end

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            data_out       <= sum[23:8];
            data_valid_out <= 1;
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule