# Python library for analysis and generation of poems in Russian #

[![Current version on PyPI](http://img.shields.io/pypi/v/rupo.svg)](https://pypi.python.org/pypi/rupo)
[![Python versions](https://img.shields.io/pypi/pyversions/rupo.svg)](https://pypi.python.org/pypi/rupo)
[![Build Status](https://travis-ci.org/IlyaGusev/rupo.svg?branch=master)](https://travis-ci.org/IlyaGusev/rupo)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rupo/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rupo)
[![Documentation Status](https://readthedocs.org/projects/rupo/badge/?version=latest)](http://rupo.readthedocs.io/en/latest/?badge=latest)

### Install ###
```
sudo pip3 install rupo
```

### Usage manual ###
#### Analysis ####
```
>>> from rupo.api import Engine
>>> engine = Engine(language="ru")
>>> engine.load()
>>> engine.get_stress("корова")
3

>>> engine.get_word_syllables("корова")
["ко", "ро", "ва"]

>>> engine.is_rhyme("корова", "здорова")
True

>>> text = "Горит восток зарёю новой.\nУж на равнине, по холмам\nГрохочут пушки. Дым багровый\nКругами всходит к небесам."
>>> engine.classify_metre(text)
iambos
```

#### Generation ####
[Model and vocabulary archive](https://www.dropbox.com/s/eefgbo53e000by5/generator_models.zip)
```
>>> from rupo.api import Engine
>>> engine = Engine(language="ru")
>>> engine.generate_poem(<LSTM model path>, <word form vocabulary path>, <gram_vectors_path>, <stress vocabulary path>, beam_width=<width of beam search>, n_syllables=<number of syllables in each line>)
<poem> or None if could't generate
```

### Models ###
* Generator: https://www.dropbox.com/s/7miw59j7mxmbyga/generator_models_v2.zip
* Stress predictor: https://www.dropbox.com/s/31pyubqskp4krsd/stress_models_light.zip
* G2P: https://www.dropbox.com/s/7rk135fzd3i8kfw/g2p_models.zip
* Dictionaries: https://www.dropbox.com/s/znqlrb1xblh3amo/dict.zip

### Литература ###
* Брейдо, 1996, [Автоматический анализ метрики русского стиха](http://search.rsl.ru/ru/record/01000000124)
* Каганов, 1996, [Лингвистическое конструирование в системах искусственного интеллекта](http://lleo.me/soft/text_dip.htm)
* Козьмин, 2006, [Автоматический анализ стиха в системе Starling](http://www.dialog-21.ru/digests/dialog2006/materials/html/Kozmin.htm)
* Гришина, 2008, [Поэтический корпус в рамках НКРЯ: общая структура и перспективы использования](http://ruscorpora.ru/sbornik2008/05.pdf)
* Пильщиков, Старостин, 2012, [Автоматическое распознавание метра: проблемы и решения](http://www.academia.edu/11465228/%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BC%D0%B5%D1%82%D1%80%D0%B0_%D0%BF%D1%80%D0%BE%D0%B1%D0%BB%D0%B5%D0%BC%D1%8B_%D0%B8_%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F)
* Барахнин, 2015, [Алгоритмы комплексного анализа русских поэтических текстов с целью автоматизации процесса создания метрических справочников и конкордансов](http://ceur-ws.org/Vol-1536/paper21.pdf), [сама система](http://poem.ict.nsc.ru/)  
