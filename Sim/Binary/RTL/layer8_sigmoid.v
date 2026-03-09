module layer8_sigmoid #(
    parameter LUT_FILE = "hex/sigmoid_lut.hex"
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg [15:0] data_out
);
    reg [7:0] lut [0:255];
    initial $readmemh(LUT_FILE, lut);

    // Clip to [-2048, +2047]
    wire signed [15:0] clipped;
    assign clipped = (data_in < -16'sh0800) ? -16'sh0800 :
                     (data_in >  16'sh07FF) ?  16'sh07FF :
                      data_in;

    // Arithmetic right-shift by 4 then centre at 128 → address [0..255]
    wire signed [15:0] addr_s;
    assign addr_s = (clipped >>> 4) + 16'sd128;
    wire [7:0] addr;
    assign addr = addr_s[7:0];

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            data_out       <= {8'd0, lut[addr]};
            data_valid_out <= 1;
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule