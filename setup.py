from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from rupo.accents.dict import AccentDict
from rupo.accents.classifier import MLAccentClassifier


class PostDevelopCommand(develop):
    def run(self):
        d = AccentDict()
        MLAccentClassifier(d)
        develop.run(self)


class PostInstallCommand(install):
    def run(self):
        d = AccentDict()
        MLAccentClassifier(d)
        install.run(self)


setup(
    name='rupo',
    packages=find_packages(),
    version='0.1.1',
    description='RuPo: library for russian poetry analysis and generation',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.1.1.tar.gz',
    keywords=['poetry', 'nlp', 'russian'],
    classifiers=[],
    package_data={
        'rupo': ['data/dict/*.txt', 'data/dict/*.trie', 'data/classifier/*.pickle'],
    },
    install_requires=[
        'dicttoxml==1.7.4',
        'datrie==0.7.1',
        'numpy==1.11.3',
        'scipy==0.18.1',
        'scikit-learn==0.18.1',
        'jsonpickle==0.9.4'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)