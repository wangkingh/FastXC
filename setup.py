from setuptools import setup, find_packages

setup(
    name='FastXC',
    version='0.1.0',
    description='A Python Package call CUDA-C and mpi command to do Cross-Corelation',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/wangkingh/FastXC',
    author='Wang Jingxi',
    author_email='1531051129@qq.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    install_requires=[
        'tqdm>=4.46.0',
    ],
    zip_safe=False,
)
