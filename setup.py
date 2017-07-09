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
    version='0.2.1',
    description='RuPo: library for russian poetry analysis and generation',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.1.4.tar.gz',
    keywords=['poetry', 'nlp', 'russian'],
    package_data={
        'rupo': ['data/dict/cmu.txt', 'data/dict/zaliznyak.txt', 'data/dict/wiki_ru.txt',
                 'data/examples/*', 'data/hyphen-tokens.txt',
                 'data/g2p_models/g2p_ru_maxlen40_BLSTM256_BLSTM256_dropout0.2_acc992_wer140.h5',
                 'data/g2p_models/ru_aligner.pickle',
                 'data/stress_models/stress_ru_word30_LSTM256_dropout0.4_acc99_wer3.h5'],
    },
    install_requires=[
        'dicttoxml>=1.7.4',
        'datrie==0.7.1',
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'jsonpickle>=0.9.4',
        'tensorflow>=1.1.0',
        'keras>=2.0.5',
        'pymorphy2>=0.8',
        'h5py>=2.7.0',
        'russian-tagsets==0.6',
        'tqdm>=4.14.0'
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