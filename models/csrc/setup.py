from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    return [
        CUDAExtension(
            name='_msmv_sampling_cuda',
            sources=[
                'msmv_sampling/msmv_sampling.cpp',
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu'
            ],
            include_dirs=['msmv_sampling'],
            extra_compile_args=dict(
                nvcc=[
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_86,code=compute_86",
                ]
            )
        )
    ]


setup(
    name='csrc',
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)

