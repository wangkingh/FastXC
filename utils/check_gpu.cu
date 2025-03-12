#include<cuda_runtime.h>
#include<stdio.h>

int main(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int n=0;n<nDevices;n++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop,n);
		printf("Device Number: %d\n",n);
		printf(" Device  name: %s\n",prop.name);
		printf(" Compute capability:%d.%d\n",prop.major,prop.minor);
	}
	return 0;
}
