from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='Installing packages following HaMeR @ https://github.com/geopavlakos/hamer',
    name='OmniHands',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        'chumpy @ git+https://github.com/mattloper/chumpy',
        'mmcv==1.3.9',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
        'plyfile',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)
