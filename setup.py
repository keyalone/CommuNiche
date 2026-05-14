from setuptools import setup, find_packages

setup(
        name="CN", 
        version='1.0',
        author="Hui-Sheng Li",
        author_email="lihs@mails.ccnu.edu.cn",
        url='https://github.com/keyalone/CommuNiche',
        description='CommuNiche enables communication-defined niches identification from spatial transcriptomics.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            ],
        classifiers= [
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.9',
)