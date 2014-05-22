

#include "OpenCLObjects.h"
#include "maxcoinMiner.h"



MaxcoinOpenCL::MaxcoinOpenCL(int _device_num) {
	this->device_num = _device_num;
	printf("Initializing GPU %d\n", device_num);
	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	printf("Initializing Device: %s\n", device->getName().c_str());

	std::vector<std::string> files_keccak;
	
	files_keccak.push_back("opencl/miner.cl");
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak);

	kernel_miner = program->getKernel("maxcoin_process");
	
	main.listDevices();

	u = device->getContext()->createBuffer(10 * sizeof(cl_ulong), CL_MEM_READ_WRITE, NULL);

	out = device->getContext()->createBuffer(sizeof(cl_int) * 255, CL_MEM_READ_WRITE, NULL);
	out_count = device->getContext()->createBuffer(sizeof(cl_int), CL_MEM_READ_WRITE, NULL);

	q = device->getContext()->createCommandQueue(device);
}


void MaxcoinOpenCL::maxcoin_process(minerMaxcoinBlock_t* block)
{
	uint32* blockInputData = (uint32*)block;

	cl_ulong targetU64 = *(cl_ulong*)(block->targetShare+24);
	OpenCLDevice* device = OpenCLMain::getInstance().getDevice(device_num);

	for(uint32 n=0; n<NUM_STEPS; n++)
	{
		if( block->height != monitorCurrentBlockHeight )
			break;
		if( (block->nTime+60) < monitorCurrentBlockTime )
		{
			block->nTime = monitorCurrentBlockTime;
		}

		kernel_miner->resetArgs();
		kernel_miner->addGlobalArg(u);
		kernel_miner->addGlobalArg(out);
		kernel_miner->addGlobalArg(out_count);
		kernel_miner->addScalarUInt(n*STEP_SIZE);
		kernel_miner->addScalarULong(targetU64);

		uint32 out_count_temp = 0;

		q->enqueueWriteBuffer(u, blockInputData, 10*sizeof(cl_ulong));
		q->enqueueWriteBuffer(out_count,&out_count_temp,sizeof(cl_uint));

		q->enqueueKernel1D(kernel_miner, STEP_SIZE,
				kernel_miner->getWorkGroupSize(device));

		q->enqueueReadBuffer(out, out_tmp, sizeof(cl_int) * 255);
		q->enqueueReadBuffer(out_count, &out_count_temp, sizeof(cl_uint));
		q->finish();

		for (int i = 0; i < out_count_temp; i++) 
		{
			totalShareCount++;
			block->nonce = out_tmp[i];
			xptMiner_submitShare(block);	
		}
		totalCollisionCount += 1 * 0x10 ; // count in steps of 0x8000 * 0x10 = 0x80000
	}
}