# -*- coding: utf-8 -*-
# Авторы: Анастасьев Даниил
# Описание: Обертка открытия больших файлов в счетчик tqdm

from contextlib import contextmanager
from os.path import getsize, basename
from tqdm import tqdm

"""
Открытие файла, обёрнутое в tqdm
"""
@contextmanager
def tqdm_open(filename, encoding='utf8'):
    total = getsize(filename)
    def wrapped_line_iterator(fd):
        with tqdm(total=total, unit="B", unit_scale=True, desc=basename(filename), miniters=1) as pb:
            processed_bytes = 0
            for line in fd:
                processed_bytes += len(line)
                if processed_bytes >= 1024 * 1024:
                    pb.update(processed_bytes)
                    processed_bytes = 0
                yield line
            pb.update(processed_bytes)

    with open(filename, encoding=encoding) as fd:
        yield wrapped_line_iterator(fd)