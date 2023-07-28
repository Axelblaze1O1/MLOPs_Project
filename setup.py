import setuptools



__version__= "0.0.0"

REPO_NAME = "MLOPs_Project"
AUTHOR_USER_NAME = "Axelblaze1O1"
SRC_REPO = "MLopsProject"
AUTHOR_EMAIL = "teotiaarnav3@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python Package for ML APP",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"" : "src"},
    packages=setuptools.find_packages(where="src")

)