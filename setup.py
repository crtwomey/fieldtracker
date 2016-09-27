
import re
from setuptools import setup

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('fieldtracker/fieldtracker.py').read(),
    re.M
).group(1)
 
 
with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")
 
 
setup(
    name = "fieldtracker",
    packages = ["fieldtracker"],
    entry_points = {
        "console_scripts": ['fieldtracker = fieldtracker.fieldtracker:main']
    },
    version = version,
    description = "Detection and tracking with python and OpenCV",
    long_description = long_descr,
    author = "Andrew Hein, Colin Twomey",
    author_email = "crtwomey@gmail.com",
    license = "BSD 3-clause open source license",
)

