from setuptools import setup, find_packages

setup(
    name="dislanguage",
    version="0.1.0",
    description=(
        "A Python toolkit to extract interpretable word-level language features "
        "for neurodegenerative disorder research (e.g., Parkinson's disease)."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="https://github.com/dangrisales/DisLanguage",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        # Core — always required
        "spacy>=3.7",
        "pyphen>=0.14",
        "pandas>=2.0",
        "tqdm>=4.0",
    ],
    extras_require={
        # f05 phonemes
        "phonemes": [
            "phonemizer>=3.2",
        ],
        # f10 morphemes
        "morphemes": [
            "morfessor>=2.0",
        ],
        # f11, f12, f21, f22, f23 — semantic variability + cluster distances
        "vectors": [
            "gensim>=4.0",
            "scipy>=1.10",
            "numpy>=1.24",
        ],
        # f16 polysemy
        "wordnet": [
            "nltk>=3.8",
        ],
        # f19 surprisal
        "surprisal": [
            "transformers>=4.30",
            "torch>=2.0",
        ],
        # Install everything at once:  pip install -e ".[all]"
        "all": [
            "phonemizer>=3.2",
            "morfessor>=2.0",
            "gensim>=4.0",
            "scipy>=1.10",
            "numpy>=1.24",
            "nltk>=3.8",
            "transformers>=4.30",
            "torch>=2.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
        "Natural Language :: Spanish",
    ],
    keywords=[
        "parkinson", "alzheimer", "neurodegenerative", "speech",
        "language", "nlp", "psycholinguistics", "clinical",
        "spanish", "feature-extraction", "disfluency",
    ],
)