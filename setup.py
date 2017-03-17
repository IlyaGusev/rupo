from distutils.core import setup
setup(
    name='rupo',
    packages=['rupo', 'rupo.accents', 'rupo.convertion', 'rupo.generate', 'rupo.main',
              'rupo.metre', 'rupo.rhymes', 'rupo.util'],
    version='0.0.1',
    description='Russian poetry analysis and generation lib',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rupo',
    download_url='https://github.com/IlyaGusev/rupo/archive/0.0.1.tar.gz',
    keywords=['poetry', 'nlp', 'russian'],
    classifiers=[],
    package_data={
        'rupo': ['data/dict/*.txt', 'data/dict/*.trie', 'data/classifier/*.pickle'],
    },
)