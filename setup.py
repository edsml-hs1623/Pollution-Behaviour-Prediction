from setuptools import setup, find_packages

setup(
    name='IRP',  # Replace with your project name
    version='0.1.0',
    author='Hanson Shen',  # Replace with your name
    author_email='hs1623@ic.ac.uk',  # Replace with your email
    description='packages needed to execute this project',
    packages=find_packages(),  # Automatically find and include packages
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'torch>=1.9.0',  # Specify your version of PyTorch
        'matplotlib>=3.4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Specify your required Python version
)
