// lcd_controller.v — HD44780 4-bit driver | 50 MHz | ML605
// Line 1: "Spl:XX Cls:Y"   Line 2: class name (NORMAL/SVEB/VEB/FUSION)

module lcd_controller (
    input  wire        clk,
    input  wire        rst,
    input  wire [4:0]  sample_num,
    input  wire [1:0]  class_out,
    output reg         LCD_RS,
    output wire        LCD_RW,
    output reg         LCD_E,
    output reg  [3:0]  LCD_DB
);
    assign LCD_RW = 1'b0;

    reg [15:0] timer = 0;
    reg        ms_tick = 0;
    always @(posedge clk) begin
        if (timer == 16'd49999) begin timer <= 0; ms_tick <= 1; end
        else                    begin timer <= timer + 1; ms_tick <= 0; end
    end

    wire [7:0] spl_tens  = 8'h30 + (sample_num / 10);
    wire [7:0] spl_units = 8'h30 + (sample_num % 10);
    wire [7:0] cls_ascii = 8'h30 + {2'b00, class_out};

    function [7:0] cls_char;
        input [1:0] cls;
        input [3:0] pos;
        begin
            case (cls)
                2'd0: case(pos) 0:"N";1:"O";2:"R";3:"M";4:"A";5:"L";default:" "; endcase
                2'd1: case(pos) 0:"S";1:"V";2:"E";3:"B";default:" "; endcase
                2'd2: case(pos) 0:"V";1:"E";2:"B";default:" "; endcase
                2'd3: case(pos) 0:"F";1:"U";2:"S";3:"I";4:"O";5:"N";default:" "; endcase
                default: cls_char = " ";
            endcase
        end
    endfunction

    reg [7:0] state = 0, wait_cnt = 0, cur_byte = 0, nxt_state = 0;
    reg [3:0] char_idx = 0;

    always @(posedge clk) begin
        if (rst) begin state <= 0; wait_cnt <= 0; char_idx <= 0; end
        else if (ms_tick) begin
            case (state)
                0:  if (wait_cnt < 20) wait_cnt <= wait_cnt + 1; else begin state <= 10; wait_cnt <= 0; end
                10: begin LCD_RS<=0; LCD_DB<=4'h3; LCD_E<=1; state<=11; end
                11: begin LCD_E<=0; state<=12; end
                12: if (wait_cnt<5) wait_cnt<=wait_cnt+1; else begin state<=13; wait_cnt<=0; end
                13: begin LCD_DB<=4'h3; LCD_E<=1; state<=14; end
                14: begin LCD_E<=0; state<=15; end
                15: if (wait_cnt<1) wait_cnt<=wait_cnt+1; else begin state<=16; wait_cnt<=0; end
                16: begin LCD_DB<=4'h3; LCD_E<=1; state<=17; end
                17: begin LCD_E<=0; state<=18; end
                18: begin LCD_DB<=4'h2; LCD_E<=1; state<=19; end
                19: begin LCD_E<=0; state<=20; end
                20: begin cur_byte<=8'h28; state<=100; nxt_state<=21; end
                21: begin cur_byte<=8'h0C; state<=100; nxt_state<=22; end
                22: begin cur_byte<=8'h06; state<=100; nxt_state<=23; end
                23: begin cur_byte<=8'h01; state<=100; nxt_state<=24; end
                24: if (wait_cnt<2) wait_cnt<=wait_cnt+1; else begin state<=30; wait_cnt<=0; end
                30: begin
                    LCD_RS <= 1;
                    case (char_idx)
                        0: cur_byte<="S"; 1: cur_byte<="p"; 2: cur_byte<="l"; 3: cur_byte<=":";
                        4: cur_byte<=spl_tens; 5: cur_byte<=spl_units;
                        6: cur_byte<=" "; 7: cur_byte<="C"; 8: cur_byte<="l"; 9: cur_byte<="s";
                        10: cur_byte<=":"; 11: cur_byte<=cls_ascii;
                        12,13,14: cur_byte<=" ";
                        default: cur_byte<=" ";
                    endcase
                    if (char_idx < 15) begin char_idx<=char_idx+1; state<=100; nxt_state<=30; end
                    else begin state<=40; char_idx<=0; end
                end
                40: begin LCD_RS<=0; cur_byte<=8'hC0; state<=100; nxt_state<=41; end
                41: begin
                    LCD_RS <= 1;
                    cur_byte <= cls_char(class_out, char_idx[3:0]);
                    if (char_idx < 12) begin char_idx<=char_idx+1; state<=100; nxt_state<=41; end
                    else state <= 50;
                end
                50: state <= 50;
                100: begin LCD_DB<=cur_byte[7:4]; LCD_E<=1; state<=101; end
                101: begin LCD_E<=0; state<=102; end
                102: begin LCD_DB<=cur_byte[3:0]; LCD_E<=1; state<=103; end
                103: begin LCD_E<=0; state<=nxt_state; end
                default: state <= 0;
            endcase
        end
    end
endmodule