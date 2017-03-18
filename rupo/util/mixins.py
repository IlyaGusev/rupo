# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Служебные миксины для удобства сериализации.


def to_dict(obj):
    """
    Преобразование объекта в словарь.

    :param obj: объект, который нужно превратить в словарь
    :return data: получившийся словарь
    """
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = to_dict(v)
        return data
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [to_dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, to_dict(value)) for key, value in obj.__dict__.items()
                    if not callable(value) and not key.startswith('_')])
        return data
    else:
        return obj


class CommonMixin(object):
    """
    Mixin для удобного сравнения и преобразования в dict.
    """
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return to_dict(self)