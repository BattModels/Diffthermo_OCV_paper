from setuptools import setup

setup(
    name='diffthermo',
    version='0.1.0',    
    description='a python package for thermodynamically consistent OCV model construction',
    url='https://github.com/BattModels/Diffthermo_OCV_paper',
    author='Archie Mingze Yao',
    author_email='amyao@umich.edu',
    license='MIT License',
    packages=['diffthermo'],
    install_requires=[
                      'numpy',       
                      'torch',
                      'pandas',
                      'scipy',
                      'matplotlib'           
                      ],

)