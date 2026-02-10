from setuptools import setup, find_packages

setup(
    name="paddleocr-genetic-sequence",
    version="1.0.0",
    py_modules=["ppstructure_genetics", "detector", "paddleocr_vl_genetics", "hybrid_genetics"],
    package_dir={"": "applications/genetic_sequence"},
    install_requires=[
        "paddleocr>=3.0.0",
        "pillow",
    ],
    python_requires=">=3.8",
)
