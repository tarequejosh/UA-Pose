from setuptools import find_packages, setup

setup(
    name='ua_pose',
    version='0.1.0',
    description='Uncertainty-Aware Lightweight Pose Estimation built on MMPose (SimCC, SOCA-PRNet, PCT)',
    packages=find_packages(include=['ua_pose', 'ua_pose.*']),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'tqdm',
        'xtcocotools',
    ],
)
