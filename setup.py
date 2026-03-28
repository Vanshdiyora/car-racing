from setuptools import setup, find_packages

setup(
    name="ppo-autonomous-driving",
    version="1.0.0",
    description="PPO Autonomous Driving with Human vs AI Racing",
    python_requires=">=3.9",
    package_dir={"": "."},
    packages=find_packages(where="."),
    entry_points={
        "console_scripts": [
            "car-racing=src.main:main",
        ],
    },
)
