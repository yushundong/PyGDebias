from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ]
)

# TODO: 完善metadata
# TODO: merge & move files (所有需要被包含在包里的, 置于PyGDebias目录下)
# TODO: 讨论接口API
# TODO: docs? metrics?
# TODO: docker image