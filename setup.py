from setuptools import setup
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='lltm',
    ext_modules=[
        DPCPPExtension('lltm_xpu', [
            'lltm_xpu.cpp',
            'lltm_xpu_kernel.cpp',
        ])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    })
