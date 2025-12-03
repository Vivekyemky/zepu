from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import sys
import os

def build_zepu():
    print("Running custom build_extension.py...")
    subprocess.check_call([sys.executable, "build_extension.py"])

class CustomBuildPy(build_py):
    def run(self):
        build_zepu()
        super().run()

class CustomDevelop(develop):
    def run(self):
        build_zepu()
        super().run()

setup(
    name="zepu",
    version="1.0.0",
    description="ZePU: Zero-overhead Processing Unit",
    author="Vivek Yemky",
    license="MIT",
    packages=find_packages(),
    cmdclass={
        'build_py': CustomBuildPy,
        'develop': CustomDevelop,
    },
    package_data={
        'zepu': ['*.dll', '*.so', '*.exe', 'zepu_worker'],
    },
    include_package_data=True,
    install_requires=[
        # Add dependencies if any, e.g. numpy, torch (optional)
    ],
)
