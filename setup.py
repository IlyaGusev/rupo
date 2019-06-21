from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)


class PostInstallCommand(install):
    def run(self):
        install.run(self)


setup(
    name='rupo',
    packages=find_packages(),
    version='0.2.8',
    description='RuPo: library for russian poetry analysis and generation',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.2.8.tar.gz',
    keywords=['poetry', 'nlp', 'russian'],
    package_data={
        'rupo': ['data/examples/*', 'data/hyphen-tokens.txt']
    },
    install_requires=[
        'dicttoxml>=1.7.4',
        'pygtrie>=2.2',
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'jsonpickle>=0.9.4',
        'pymorphy2>=0.8',
        'h5py>=2.7.0',
        'russian-tagsets==0.6',
        'tqdm>=4.14.0',
        'rnnmorph==0.2.3',
        'sentence_splitter>=1.2',
        'rulm==0.0.2',
        'russ==0.0.1'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
