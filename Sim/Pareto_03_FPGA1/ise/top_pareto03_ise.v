// top_pareto03_ise.v — ISE Top-Level | ML605 Virtex-6 | Pareto-03 ECG CNN
// Clock   : Clocking Wizard (200MHz LVDS → 50MHz) — generate via CORE Generator
// Control : sw[4:0] DIP switches select test sample 0-29
// Output  : led[3:0] one-hot class, LCD shows sample + class name

module top_pareto03_ise (
    input  wire        sys_clk_p,
    input  wire        sys_clk_n,
    input  wire        reset,
    input  wire [4:0]  sw,
    output wire [3:0]  led,
    output wire        LCD_RS,
    output wire        LCD_RW,
    output wire        LCD_E,
    output wire [3:0]  LCD_DB
);
    // 1. Clocking Wizard — generate "clk_wiz" in CORE Generator
    //    Input: CLK_IN1_D differential 200MHz | Output: CLK_OUT1 50MHz
    wire clk, clk_locked;
    clk_wiz u_clk_wiz (
        .CLK_IN1_P(sys_clk_p), .CLK_IN1_N(sys_clk_n),
        .CLK_OUT1(clk), .LOCKED(clk_locked), .RESET(1'b0)
    );

    // 2. Synchronised reset (held until wizard locks)
    reg rst_r0, rst_sync;
    always @(posedge clk or negedge clk_locked) begin
        if (!clk_locked) begin rst_r0 <= 1; rst_sync <= 1; end
        else             begin rst_r0 <= reset; rst_sync <= rst_r0; end
    end
    wire rst = rst_sync;

    // 3. Test sample ROM — all 30 ECG samples in one BRAM
    wire        ecg_valid;
    wire signed [15:0] ecg_data;
    wire        stream_done;
    test_all_samples_rom #(
        .INIT_FILE("ise/combined_test_inputs.hex"),
        .SAMPLE_LEN(180), .N_SAMPLES(30)
    ) u_rom (
        .clk(clk), .reset(rst), .sw(sw),
        .data_valid_out(ecg_valid), .data_out(ecg_data), .done(stream_done)
    );

    // 4. CNN inference pipeline
    wire        cnn_valid;
    wire [15:0] cnn_result;
    top_pareto03 u_cnn (
        .clk(clk), .reset(rst),
        .data_valid_in(ecg_valid), .data_in(ecg_data),
        .data_valid_out(cnn_valid), .data_out(cnn_result)
    );

    // 5. Latch result
    reg [1:0] class_hold;
    reg [4:0] sample_hold;
    always @(posedge clk) begin
        if (rst) begin class_hold <= 0; sample_hold <= sw; end
        else begin
            sample_hold <= sw;
            if (cnn_valid) class_hold <= cnn_result[1:0];
        end
    end

    // 6. LEDs — one-hot: [0]=Normal [1]=SVEB [2]=VEB [3]=Fusion
    assign led[0] = (class_hold == 2'd0);
    assign led[1] = (class_hold == 2'd1);
    assign led[2] = (class_hold == 2'd2);
    assign led[3] = (class_hold == 2'd3);

    // 7. LCD
    lcd_controller u_lcd (
        .clk(clk), .rst(rst),
        .sample_num(sample_hold), .class_out(class_hold),
        .LCD_RS(LCD_RS), .LCD_RW(LCD_RW),
        .LCD_E(LCD_E), .LCD_DB(LCD_DB)
    );
endmodule
