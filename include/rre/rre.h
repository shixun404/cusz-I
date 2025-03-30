#pragma once

void TCMS_COMPRESS(void* input, size_t insize, uint8_t** output, size_t* outsize, size_t* tcms_padding_bytes, float* time);
void TCMS_DECOMPRESS(uint8_t* input, void** output, size_t tcms_padding_bytes, float* time);
void BITR_COMPRESS(void* input, size_t insize, uint8_t** output, size_t* outsize, size_t* bitr_padding_bytes, float* time);
void BITR_DECOMPRESS(uint8_t* input, void** output, size_t bitr_padding_bytes, float* time);
