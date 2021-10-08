from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="rnntLoss",
    version="1.0",
    ext_modules=[
        CUDAExtension(
            name="rnntLoss",
            sources=["rnntLoss.cpp", "rnnt.cu"]
            )
        ],
    cmdclass={"build_ext": BuildExtension}
    )


