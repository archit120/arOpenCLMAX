#pragma comment(lib,"Ws2_32.lib")
#include<Winsock2.h>
#include<ws2tcpip.h>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>


#include"jhlib.h" // slim version of jh library

#include "OpenCLObjects.h"

// connection info for xpt
typedef struct  
{
	char* ip;
	uint16 port;
	char* authUser;
	char* authPass;
}generalRequestTarget_t;

#include"xptServer.h"
#include"xptClient.h"

#include"sha2.h"

#include"transaction.h"

// global settings for miner
typedef struct  
{
	generalRequestTarget_t requestTarget;
	uint32 protoshareMemoryMode;
	// GPU
	bool useGPU; // enable OpenCL
	// GPU (MaxCoin specific)

}minerSettings_t;

extern minerSettings_t minerSettings;

#define PROTOSHARE_MEM_512		(0)
#define PROTOSHARE_MEM_256		(1)
#define PROTOSHARE_MEM_128		(2)
#define PROTOSHARE_MEM_32		(3)
#define PROTOSHARE_MEM_8		(4)

// block data struct

typedef struct  
{
	// block header data (relevant for midhash)
	uint32	version;
	uint8	prevBlockHash[32];
	uint8	merkleRoot[32];
	uint32	nTime;
	uint32	nBits;
	uint32	nonce;
	// birthday collision
	uint32	birthdayA;
	uint32	birthdayB;
	uint32	uniqueMerkleSeed;

	uint32	height;
	uint8	merkleRootOriginal[32]; // used to identify work
	uint8	target[32];
	uint8	targetShare[32];
}minerProtosharesBlock_t;

typedef struct  
{
	// block header data
	uint32	version;
	uint8	prevBlockHash[32];
	uint8	merkleRoot[32];
	uint32	nTime;
	uint32	nBits;
	uint32	nonce;
	uint32	uniqueMerkleSeed;
	uint32	height;
	uint8	merkleRootOriginal[32]; // used to identify work
	uint8	target[32];
	uint8	targetShare[32];
}minerScryptBlock_t;

typedef struct  
{
	// block header data
	uint32	version;
	uint8	prevBlockHash[32];
	uint8	merkleRoot[32];
	uint32	nTime;
	uint32	nBits;
	uint32	nonce;
	uint32	uniqueMerkleSeed;
	uint32	height;
	uint8	merkleRootOriginal[32]; // used to identify work
	uint8	target[32];
	uint8	targetShare[32];
	// found chain data
	// todo
}minerPrimecoinBlock_t;

typedef struct  
{
	// block data (order and memory layout is important)
	uint32	version;
	uint8	prevBlockHash[32];
	uint8	merkleRoot[32];
	uint32	nTime;
	uint32	nBits;
	uint32	nonce;
	// remaining data
	uint32	uniqueMerkleSeed;
	uint32	height;
	uint8	merkleRootOriginal[32]; // used to identify work
	uint8	target[32];
	uint8	targetShare[32];
}minerMetiscoinBlock_t; // identical to scryptBlock

typedef struct  
{
	// block data (order and memory layout is important)
	uint32	version;
	uint8	prevBlockHash[32];
	uint8	merkleRoot[32];
	uint32	nTime;
	uint32	nBits;
	uint32	nonce;
	// remaining data
	uint32	uniqueMerkleSeed;
	uint32	height;
	uint8	merkleRootOriginal[32]; // used to identify work
	uint8	target[32];
	uint8	targetShare[32];
}minerMaxcoinBlock_t; // identical to scryptBlock

#include"scrypt.h"
#include"algorithm.h"

void xptMiner_submitShare(minerProtosharesBlock_t* block);
void xptMiner_submitShare(minerScryptBlock_t* block);
void xptMiner_submitShare(minerPrimecoinBlock_t* block);
void xptMiner_submitShare(minerMetiscoinBlock_t* block);
void xptMiner_submitShare(minerMaxcoinBlock_t* block);

// stats
extern volatile uint32 totalCollisionCount;
extern volatile uint32 totalShareCount;
extern volatile uint32 totalRejectedShareCount;

extern volatile uint32 monitorCurrentBlockHeight;
extern volatile uint32 monitorCurrentBlockTime;