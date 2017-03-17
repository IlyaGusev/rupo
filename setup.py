from setuptools import find_packages, setup
setup(
    name='rupo',
    packages=find_packages(),
    version='0.0.2',
    description='RuPo: library for russian poetry analysis and generation',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.0.2.tar.gz',
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
)