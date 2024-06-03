from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extension_mod = CUDAExtension(
        name='fastgl', 
        sources=[   
                    'cuda_operators/aggregate_acc.cpp', 
                    'cuda_operators/aggregate_acc_kernel.cu',
                    'cuda_operators/id_map.cu',
                    'cuda_operators/sample.cu',
                    'cuda_operators/utilis.cu'
                ],
        extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )

# Specify the CUDA architecture flags
CUDA_ARCH_FLAGS = ['-gencode', 'arch=compute_80,code=sm_80']
extension_mod.extra_compile_args['nvcc'] = CUDA_ARCH_FLAGS

setup(
    name='fastgl',
    ext_modules=[
        extension_mod
    ],
    cmdclass={
        'build_ext': BuildExtension
    })