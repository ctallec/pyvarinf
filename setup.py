from setuptools import setup

main_version = '0'
subversion = '1'

version = main_version + '.' + subversion
setup(name='pyvarinf',
      version=version,
      url='https://gitlab.inria.fr/ctallec/VarInf',
      author='Corentin Tallec',
      author_email='corentinxtallec@gmail.com',
      license='MIT',
      packages=['pyvarinf'],
      zip_safe=False)

