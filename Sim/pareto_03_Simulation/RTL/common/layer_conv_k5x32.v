module layer_conv_k5x32 #(
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = "",
    parameter FILTER_ID   = 0,
    parameter STREAM_LENGTH = 45
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [511:0] data_in, // 32 channels * 16 bits
    output reg data_valid_out,
    output reg signed [31:0] data_out
);
    reg signed [15:0] window  [0:31][0:4];
    // Weights shape: [16 filters, 32 channels, 1, 5] -> 16*32*5 = 2560 elements
    reg signed [15:0] weights [0:2559];
    reg signed [15:0] biases  [0:15];
    reg [3:0] count;
    reg [1:0] flush_count;
    reg [7:0] element_count;
    
    reg signed [31:0] sum;
    integer c, m;
    
    always @(*) begin
        sum = {{8{biases[FILTER_ID][15]}}, biases[FILTER_ID], 8'd0}; 
        
        for (c = 0; c < 32; c = c + 1) begin
            for (m = 0; m < 5; m = m + 1) begin
                // 3D flattening to 1D: TFLite shape is [out_ch, 1, kernel, in_ch]
                // index = FILTER_ID*(5*32) + m*32 + c
                sum = sum + (window[c][4 - m] * weights[FILTER_ID*160 + m*32 + c]);
            end
        end
    end

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (BIAS_FILE != "")   $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            for (c = 0; c < 32; c = c + 1) begin
                window[c][0] <= 0; window[c][1] <= 0; window[c][2] <= 0;
                window[c][3] <= 0; window[c][4] <= 0;
            end
            count          <= 0;
            flush_count    <= 0;
            element_count  <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            for (c = 0; c < 32; c = c + 1) begin
                window[c][4] <= window[c][3];
                window[c][3] <= window[c][2];
                window[c][2] <= window[c][1];
                window[c][1] <= window[c][0];
                window[c][0] <= data_in[c*16 +: 16];
            end

            if (count < 5) count <= count + 1;

            if (count >= 3) begin
                data_valid_out <= 1'b1;
                data_out       <= sum;
            end else begin
                data_valid_out <= 1'b0;
            end
            
            element_count <= element_count + 1;
            if (element_count == STREAM_LENGTH - 1) begin
                flush_count <= 3;
            end
        end else if (flush_count > 0) begin
            for (c = 0; c < 32; c = c + 1) begin
                window[c][4] <= window[c][3];
                window[c][3] <= window[c][2];
                window[c][2] <= window[c][1];
                window[c][1] <= window[c][0];
                window[c][0] <= 0;
            end
            
            data_valid_out <= 1'b1;
            data_out       <= sum;
            flush_count    <= flush_count - 1;
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule
