export ASSET_PATH=/home/vgi-robot/Working/FetchBench-CORL2024
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

##cu117 setting
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDACXX=$CUDA_HOME/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.6+PTX"

##Vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$(dirname "$0")/../InfiniGym"
##CGN
#python -m isaacgymenvs.eval task=FetchMeshCuroboPtdCGNBeta scene=benchmark_eval/RigidObjDesk_0

##GraspGen
python -m isaacgymenvs.eval task=FetchMeshCuroboGraspGen scene=benchmark_eval/RigidObjDesk_0
