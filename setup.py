from setuptools import find_packages, setup

setup(
    name="ai-clips-maker",
    py_modules=["ai_clips_maker"],
    version="1.0.0",
    description=(
        "AI Clips Maker is an open-source Python library that automatically converts "
        "long videos into concise, speaker-aware clips using face and voice detection."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alperen Sümeroğlu",
    author_email="alperennsumeroglu@gmail.com",
    url="https://github.com/alperensumeroglu/ai-clips-maker",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "av",
        "facenet-pytorch",
        "matplotlib",
        "mediapipe",
        "nltk",
        "numpy",
        "opencv-python",
        "pandas",
        "psutil",
        "pyannote.audio",
        "pyannote.core",
        "pynvml",
        "pytest",
        "python-magic",
        "scenedetect",
        "scikit-learn",
        "sentence-transformers",
        "scipy",
        "torch",
    ],
    zip_safe=False,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/alperensumeroglu/ai-clips-maker#readme",
        "Homepage": "https://github.com/alperensumeroglu/ai-clips-maker",
        "Repository": "https://github.com/alperensumeroglu/ai-clips-maker",
        "Issues": "https://github.com/alperensumeroglu/ai-clips-maker/issues",
    },
    include_package_data=True,
    extras_require={
        "dev": [
            "black",
            "black[jupyter]",
            "build",
            "flake8",
            "ipykernel",
            "pytest",
            "twine",
        ],
    },
)
