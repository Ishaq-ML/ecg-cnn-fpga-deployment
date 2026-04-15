/**
 ******************************************************************************
 * @file    ads1115.h
 * @brief   ADS1115 16-bit ADC driver for ECG acquisition via I2C.
 *
 * Hardware: ADS1115 ADDR pin → GND → I2C address 0x48
 * Channel:  AIN0 single-ended (ECG signal from AD8232 output)
 * PGA:      ±4.096 V  (covers full AD8232 output range on 3.3 V supply)
 * Mode:     Continuous conversion, 250 SPS
 *
 * LSB resolution: 4.096 V / 32768 = 125 µV  →  0.125 mV per count
 ******************************************************************************
 */

#ifndef ADS1115_H
#define ADS1115_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32g0xx_hal.h"

/* -------------------------------------------------------------------------- */
/* I2C address (ADDR pin tied to GND → 0x48, shifted left for HAL)           */
/* -------------------------------------------------------------------------- */
#define ADS1115_I2C_ADDR        (0x48U << 1U)   /* 0x90 */
#define ADS1115_I2C_TIMEOUT_MS  100U

/* -------------------------------------------------------------------------- */
/* Internal register pointers                                                 */
/* -------------------------------------------------------------------------- */
#define ADS1115_REG_CONVERSION  0x00U
#define ADS1115_REG_CONFIG      0x01U
#define ADS1115_REG_LO_THRESH   0x02U
#define ADS1115_REG_HI_THRESH   0x03U

/* -------------------------------------------------------------------------- */
/* Config register (16-bit, MSB first)                                        */
/*                                                                             */
/* Bit 15    OS   : 0 – no effect in continuous mode                          */
/* Bits 14:12 MUX : 100 – AIN0 vs GND (single-ended)                         */
/* Bits 11:9  PGA : 001 – ±4.096 V                                            */
/* Bit  8    MODE : 0   – continuous conversion                               */
/* Bits  7:5  DR  : 101 – 250 SPS                                             */
/* Bit   4   COMP_MODE : 0 – traditional comparator                           */
/* Bit   3   COMP_POL  : 0 – active low                                       */
/* Bit   2   COMP_LAT  : 0 – non-latching                                     */
/* Bits  1:0 COMP_QUE  : 11 – disable comparator / ALRT pin                  */
/*                                                                             */
/*  0 100 001 0 101 0 0 0 11  =  0b 0100 0010 1010 0011  =  0x42A3           */
/* -------------------------------------------------------------------------- */
#define ADS1115_CONFIG_VALUE    0x42A3U

/* LSB size in millivolts (PGA = ±4.096 V  →  4.096 / 32768 V = 0.125 mV)  */
#define ADS1115_LSB_MV          0.125f

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief  Configure the ADS1115 for continuous ECG acquisition.
 * @param  hi2c  Pointer to an STM32 HAL I2C handle.
 * @retval HAL_OK on success, HAL_ERROR / HAL_TIMEOUT on failure.
 */
HAL_StatusTypeDef ADS1115_Init(I2C_HandleTypeDef *hi2c);

/**
 * @brief  Read the latest 16-bit signed conversion result.
 * @param  hi2c  Pointer to an STM32 HAL I2C handle.
 * @retval Signed 16-bit ADC value (two's complement).
 *         Returns 0 on I2C error.
 */
int16_t ADS1115_ReadRaw(I2C_HandleTypeDef *hi2c);

/**
 * @brief  Convert a raw ADS1115 reading to millivolts.
 * @param  raw  Signed 16-bit value from ADS1115_ReadRaw().
 * @retval Voltage in mV (float).
 */
float ADS1115_ToMillivolts(int16_t raw);

#ifdef __cplusplus
}
#endif

#endif /* ADS1115_H */
