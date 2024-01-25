# Python library for analysis and generation of poems in Russian #

[![Current version on PyPI](http://img.shields.io/pypi/v/rupo.svg)](https://pypi.python.org/pypi/rupo)
[![Python versions](https://img.shields.io/pypi/pyversions/rupo.svg)](https://pypi.python.org/pypi/rupo)
[![Build Status](https://travis-ci.org/IlyaGusev/rupo.svg?branch=master)](https://travis-ci.org/IlyaGusev/rupo)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rupo/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rupo)
[![Documentation Status](https://readthedocs.org/projects/rupo/badge/?version=latest)](http://rupo.readthedocs.io/en/latest/?badge=latest)

### Install ###
Warning: Python 3.9+ is not supported! Use Python 3.8.

```
git clone https://github.com/IlyaGusev/rupo
cd rupo
pip install -r requirements.txt
sh download.sh
```

### Example ###
https://colab.research.google.com/drive/1WBl9erJvC9Oc9PjCD8JyC_40TDUqahCx

### Usage manual ###
#### Analysis ####
```
>>> from rupo.api import Engine
>>> engine = Engine(language="ru")
>>> engine.load(<stress model path>, <zalyzniak dict path>)
>>> engine.get_stresses("корова")
[3]

>>> engine.get_word_syllables("корова")
["ко", "ро", "ва"]

>>> engine.is_rhyme("корова", "здорова")
True

>>> text = "Горит восток зарёю новой.\nУж на равнине, по холмам\nГрохочут пушки. Дым багровый\nКругами всходит к небесам."
>>> engine.classify_metre(text)
iambos
```

#### Generation ####
Script for poem generation. It can work in two different modes: sampling or beam search.

```
python generate_poem.py
```

| Argument            | Default | Description                                |
|:--------------------|:--------|:-------------------------------------------|
| --metre-schema      | +-      | feet type: -+ (iambos), +- (trochee), ...  |
| --rhyme-pattern     | abab    | rhyme pattern                              |
| --n-syllables       | 8       | number of syllables in line                |
| --sampling-k        | 50000   | top-k words to sample from (sampling mode) |
| --beam-width        | None    | width of beam search (beam search mode)    |
| --temperature       | 1.0     | sampling softmax temperature               |
| --last-text         | None    | custom last line                           |
| --count             | 100     | count of poems to generate                 |
| --model-path        | None    | optional path to generator model directory |
| --token-vocab-path  | None    | optional path to vocabulary                |
| --stress-vocab-path | None    | optional path to stress vocabulary         |

## Models ###
* Generator: https://www.dropbox.com/s/dwkui2xqivzsyw5/generator_model.zip
* Stress predictor: https://www.dropbox.com/s/i9tarc8pum4e40p/stress_models_14_05_17.zip
* G2P: https://www.dropbox.com/s/7rk135fzd3i8kfw/g2p_models.zip
* Dictionaries: https://www.dropbox.com/s/znqlrb1xblh3amo/dict.zip

### Литература ###
* Брейдо, 1996, [Автоматический анализ метрики русского стиха](http://search.rsl.ru/ru/record/01000000124)
* Каганов, 1996, [Лингвистическое конструирование в системах искусственного интеллекта](http://lleo.me/soft/text_dip.htm)
* Козьмин, 2006, [Автоматический анализ стиха в системе Starling](http://www.dialog-21.ru/digests/dialog2006/materials/html/Kozmin.htm)
* Гришина, 2008, [Поэтический корпус в рамках НКРЯ: общая структура и перспективы использования](http://ruscorpora.ru/sbornik2008/05.pdf)
* Пильщиков, Старостин, 2012, [Автоматическое распознавание метра: проблемы и решения](http://www.academia.edu/11465228/%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BC%D0%B5%D1%82%D1%80%D0%B0_%D0%BF%D1%80%D0%BE%D0%B1%D0%BB%D0%B5%D0%BC%D1%8B_%D0%B8_%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F)
* Барахнин, 2015, [Алгоритмы комплексного анализа русских поэтических текстов с целью автоматизации процесса создания метрических справочников и конкордансов](http://ceur-ws.org/Vol-1536/paper21.pdf), [сама система](http://poem.ict.nsc.ru/)
