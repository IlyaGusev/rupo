from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    def run(self):
        from rupo.stress.dict import StressDict
        from rupo.stress.stress_classifier import MLStressClassifier
        d = StressDict()
        MLStressClassifier(d)
        develop.run(self)


class PostInstallCommand(install):
    def run(self):
        from rupo.stress.dict import StressDict
        from rupo.stress.stress_classifier import MLStressClassifier
        d = StressDict()
        MLStressClassifier(d)
        install.run(self)


setup(
    name='rupo',
    packages=find_packages(),
    version='0.1.4',
    description='RuPo: library for russian poetry analysis and generation',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.1.4.tar.gz',
    keywords=['poetry', 'nlp', 'russian'],
    package_data={
        'rupo': ['data/dict/*.txt', 'data/examples/*'],
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