import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='meta-self-learner',
    version='0.0.3',
    scripts=['meta_self_learner_package'],
    install_requires=[
        "pandas",
        "numpy",
        "tabulate",
        "xgboost",
        "scipy",
        "six",
        "scikit_learn",
        "matplotlib",
        "plot-metric",
    ],
    author="Ajay Arunachalam",
    author_email="ajay.arunachalam08@gmail.com",
    description="Meta Ensemble Self-Learning with Optimization Objective Functions",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/ajayarunachalam/meta-self-learner/',
    packages=setuptools.find_packages(),
    py_modules=['msl/MetaLearning', 'msl/cf_matrix'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
