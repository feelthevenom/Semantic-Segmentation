from setuptools import setup, find_packages
from typing import List


def get_requirements()->List[str]:

    requirements_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as f:

            lines = f.readlines()

            for line in lines:
                requirements = line.strip()
                if requirements and requirements!='-e .':
                    requirements_list.append(requirements)
        return requirements_list
    except FileNotFoundError as e:
        print(f"{e} file not found Error occored in Setup.py")


setup(
    name = "Semantic Segmentation",
    version="0.0.1",
    author="Rahul Ravikumar",
    author_email="rahul.valli2003@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)