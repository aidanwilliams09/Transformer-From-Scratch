import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='transformer',
    version='0.0.4',
    author='552 Group 2',
    author_email='aidan.williams@mail.mcgill.ca',
    description='Transformer model for weather prediction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aidanwilliams09/Transformer-From-Scratch',
    license='MIT',
    packages=['transformer_552'],
    install_requires=[
        'torch',
    ],
)