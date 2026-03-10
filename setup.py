from setuptools import setup, find_packages

setup(
    name="dct_nic",
    version="1.0.0",
    description="DCT Basis Benchmarks for Neural Image Compression",
    author="Nikolay Kalmykov et al.",
    url="https://github.com/nkalmykovsk/dct_benchmark_nic",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "pillow>=9.0",
        "imageio>=2.31",
    ],
    extras_require={
        "compressai": ["compressai>=1.2"],
        "perceptual": ["lpips>=0.1.4", "pytorch-msssim>=1.0"],
        "jxl": ["imagecodecs"],
        "notebooks": ["jupyter", "ipywidgets"],
        "all": [
            "compressai>=1.2",
            "lpips>=0.1.4",
            "pytorch-msssim>=1.0",
            "imagecodecs",
            "jupyter",
        ],
    },
)
