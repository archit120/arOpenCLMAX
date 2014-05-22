#ifndef __MAXCOIN_MINER_H__
#define __MAXCOIN_MINER_H__
#include "global.h"

#define STEP_SIZE 0x80000
#define NUM_STEPS 0x100
#define STEP_MULTIPLIER 0x10000

class MaxcoinOpenCL {
public:

	MaxcoinOpenCL(int device_num);
	void maxcoin_process(minerMaxcoinBlock_t* block);
private:

	int device_num;

	OpenCLKernel* kernel_miner;

	OpenCLBuffer* u;
	OpenCLBuffer* out;
	OpenCLBuffer* out_count;
	OpenCLCommandQueue * q;
	uint64 out_tmp[255];
};

#endif