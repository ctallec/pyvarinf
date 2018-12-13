""" Setup file pyvarinf """
from setuptools import setup

main_version = '0'
subversion = '2'

version = main_version + '.' + subversion
setup(name='pyvarinf',
      version=version,
      url='https://github.com/ctallec/pyvarinf',
      download_url='https://github.com/ctallec/pyvarinf/archive/0.1.tar.gz',
      author='Corentin Tallec, Leonard Blier',
      author_email='corentinxtallec@gmail.com',
      license='MIT',
      packages=['pyvarinf'],
      zip_safe=False)
