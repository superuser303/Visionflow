from setuptools import setup, find_packages

setup(
    name="visionflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "visionflow=scripts.run_webcam:main",
        ]
    },
)