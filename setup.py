import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentiment", # Replace with your own username
    version="0.0.1",
    author="Fred Commo",
    author_email="fcommo@kpmg.fr",
    description="Combine multiple pretrained models to run sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        # "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: KPMG License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)