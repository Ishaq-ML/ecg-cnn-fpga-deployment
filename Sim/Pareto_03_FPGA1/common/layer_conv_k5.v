module layer_conv_k5 #(
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = "",
    parameter FILTER_ID   = 0,
    parameter STREAM_LENGTH = 180
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg signed [31:0] data_out // Outputting 32-bit before BN to keep precision
);
    reg signed [15:0] window  [0:4];
    reg signed [15:0] weights [0:79]; // Supports up to 16 input channels (16 * 5)
    reg signed [15:0] biases  [0:63]; // Supports up to 64 filters
    reg [3:0] count;
    reg [1:0] flush_count;
    reg [7:0] element_count;
    reg start_flush;
    
    // Total sum needs to sum the MACs of all input channels for this filter
    // For simplicity of a generic module, we receive serialized 1-channel inputs at a time, 
    // but the system design implies spatial sliding if channels > 1. 
    // For single channel (Block 1):
    reg signed [31:0] sum;
    integer m;
    
    always @(*) begin
        // Start with the Bias (shifted for Q8.8 -> Q16.16 equivalence if needed)
        // Since input is Q8.8 and weight is Q8.8, MAC result will be Q16.16. 
        // Bias is Q8.8 but actually imported as INT32 sometimes? 
        // Based on model_structure.txt, bias is Q8.8 INT32 dequant.
        // We assume it's scaled properly to add directly to the MAC result.
        sum = {{8{biases[FILTER_ID][15]}}, biases[FILTER_ID], 8'd0}; // Basic bias add shifted to Q16.16
        
        for (m = 0; m < 5; m = m + 1) begin
            // We use standard 16-bit * 16-bit = 32-bit arithmetic.
            sum = sum + (window[4 - m] * weights[FILTER_ID*5 + m]);
        end
    end

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (BIAS_FILE != "")   $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            window[0]      <= 0; window[1] <= 0; window[2] <= 0;
            window[3]      <= 0; window[4] <= 0;
            count          <= 0;
            flush_count    <= 0;
            element_count  <= 0;
            start_flush    <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            // Shift register
            window[4] <= window[3];
            window[3] <= window[2];
            window[2] <= window[1];
            window[1] <= window[0];
            window[0] <= data_in;

            if (count < 5) count <= count + 1;

            // Valid after 5 elements enter the window (for Valid pad)
            // But for Same pad, we want valid after 3 elements
            if (count >= 3) begin
                data_valid_out <= 1'b1;
                data_out       <= sum; // Raw 32-bit output passed to BN
            end else begin
                data_valid_out <= 1'b0;
            end
            
            element_count <= element_count + 1;
            if (element_count == STREAM_LENGTH - 1) begin
                flush_count <= 3;
            end
        end else if (flush_count > 0) begin
            window[4] <= window[3];
            window[3] <= window[2];
            window[2] <= window[1];
            window[1] <= window[0];
            window[0] <= 0;

            data_valid_out <= 1'b1;
            data_out       <= sum;
            flush_count    <= flush_count - 1;
        end else begin
            data_valid_out <= 1'b0;
            // Removed count <= 0 to prevent resetting mid-stream during stalls
        end
    end
endmodule
