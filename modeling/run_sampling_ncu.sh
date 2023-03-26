/usr/local/cuda/bin/ncu --section "regex:MemoryWorkloadAnalysis" -o test -f --nvtx --nvtx-include "sampling]" python3 modeling/sampling_ncu.py 

echo "system read in sector [32 bytes]"
/usr/local/cuda/bin/ncu -i test.ncu-rep --page raw | grep "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum"
echo " "

echo "dram read in sector [32 bytes]"
/usr/local/cuda/bin/ncu -i test.ncu-rep --page raw | grep "dram__sectors_read.sum"
echo " "

echo "dram write in sector [32 bytes]"
/usr/local/cuda/bin/ncu -i test.ncu-rep --page raw | grep "dram__sectors_write.sum"
echo " "