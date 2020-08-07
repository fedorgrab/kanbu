from setuptools import find_packages, setup


def read_reqs(req_file: str):
    with open(req_file) as req:
        return [
            line.strip()
            for line in req.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="kanbu",
    packages=find_packages(),
    author="Fedor Grab",
    author_email="fvgrab@gmail.com",
    description="A simple library consisting of data processing utils",
    license="MIT",
    install_requires=read_reqs("requirements.txt"),
    include_package_data=True,
    version="0.1",
    zip_safe=False,
)
