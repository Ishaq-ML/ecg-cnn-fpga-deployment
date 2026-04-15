/**
 ******************************************************************************
 * @file    ads1115.c
 * @brief   ADS1115 16-bit I2C ADC driver – continuous ECG acquisition.
 ******************************************************************************
 */

#include "ads1115.h"

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief  Write a 16-bit value to an ADS1115 register.
 */
static HAL_StatusTypeDef ADS1115_WriteReg(I2C_HandleTypeDef *hi2c,
                                           uint8_t reg,
                                           uint16_t value)
{
    uint8_t buf[3];
    buf[0] = reg;                             /* Register pointer  */
    buf[1] = (uint8_t)((value >> 8U) & 0xFFU); /* MSB first         */
    buf[2] = (uint8_t)(value & 0xFFU);          /* LSB               */

    return HAL_I2C_Master_Transmit(hi2c,
                                   ADS1115_I2C_ADDR,
                                   buf, 3U,
                                   ADS1115_I2C_TIMEOUT_MS);
}

/**
 * @brief  Set the ADS1115 register pointer (without writing data).
 */
static HAL_StatusTypeDef ADS1115_SetPointer(I2C_HandleTypeDef *hi2c,
                                              uint8_t reg)
{
    return HAL_I2C_Master_Transmit(hi2c,
                                   ADS1115_I2C_ADDR,
                                   &reg, 1U,
                                   ADS1115_I2C_TIMEOUT_MS);
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

HAL_StatusTypeDef ADS1115_Init(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef status;

    /* Write configuration: AIN0 single-ended, PGA ±4.096 V,
       continuous mode, 250 SPS, comparator disabled.               */
    status = ADS1115_WriteReg(hi2c, ADS1115_REG_CONFIG, ADS1115_CONFIG_VALUE);
    if (status != HAL_OK)
    {
        return status;
    }

    /* Pre-point the address pointer at the conversion register so
       subsequent reads only require an I2C receive (no register
       pointer write each time, halves the I2C bus traffic).        */
    status = ADS1115_SetPointer(hi2c, ADS1115_REG_CONVERSION);
    if (status != HAL_OK)
    {
        return status;
    }

    /* Wait for the first conversion to complete (250 SPS → 4 ms)  */
    HAL_Delay(8U);

    return HAL_OK;
}

int16_t ADS1115_ReadRaw(I2C_HandleTypeDef *hi2c)
{
    uint8_t buf[2] = {0U, 0U};

    /* In continuous mode the pointer is already set to 0x00 after
       Init; just do a 2-byte read.                                 */
    if (HAL_I2C_Master_Receive(hi2c,
                                ADS1115_I2C_ADDR,
                                buf, 2U,
                                ADS1115_I2C_TIMEOUT_MS) != HAL_OK)
    {
        return 0;   /* Return 0 on bus error rather than garbage     */
    }

    /* ADS1115 sends MSB first (big-endian)                         */
    return (int16_t)((uint16_t)buf[0] << 8U | (uint16_t)buf[1]);
}

float ADS1115_ToMillivolts(int16_t raw)
{
    /* PGA = ±4.096 V  →  1 LSB = 0.125 mV                        */
    return (float)raw * ADS1115_LSB_MV;
}
