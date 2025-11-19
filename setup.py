from setuptools import setup, find_packages

setup(
    name="inductor-debug-module",
    version="0.1.0",
    py_modules=["mock_backend"],
    install_requires=["torch>=2.0.0"],
)
