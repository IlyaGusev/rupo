# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты к классификатору метра.

import unittest
import jsonpickle

from rupo.main.markup import Markup
from rupo.stress.predictor import CombinedStressPredictor
from rupo.metre.metre_classifier import MetreClassifier, ClassificationResult, StressCorrection


class TestMetreClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_predictor = CombinedStressPredictor()

    def test_classification_result(self):
        result = ClassificationResult(5)
        result.additions["iambos"].append(StressCorrection(0, 0, 0, "", 0))
        self.assertEqual(result, jsonpickle.decode(result.to_json()))

    def test_metre_classifier1(self):
        text = "Горит восток зарёю новой.\n" \
               "Уж на равнине, по холмам\n" \
               "Грохочут пушки. Дым багровый\n" \
               "Кругами всходит к небесам."
        markup, result = MetreClassifier.improve_markup(Markup.process_text(text, self.stress_predictor))
        self.assertIsInstance(markup, Markup)
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.metre, "iambos")

    def test_metre_classifier2(self):
        text = "Буря мглою небо кроет,\n" \
               "Вихри снежные крутя;\n" \
               "То, как зверь, она завоет,\n" \
               "То заплачет, как дитя..."
        markup, result = MetreClassifier.improve_markup(Markup.process_text(text, self.stress_predictor))
        self.assertEqual(result.metre, "choreios")

    def test_metre_classifier3(self):
        text = "На стеклах нарастает лед,\n"\
               "Часы твердят: «Не трусь!»\n"\
               "Услышать, что ко мне идет,\n"\
               "И мертвой я боюсь.\n"\
               "Как идола, молю я дверь;\n"\
               "«Не пропускай беду!»\n"\
               "Кто воет за стеной, как зверь,\n"\
               "Кто прячется в саду?"
        markup, result = MetreClassifier.improve_markup(Markup.process_text(text, self.stress_predictor))
        self.assertEqual(result.metre, "iambos")

    def test_metre_classifier4(self):
        text = "Вот уж вечер. Роса\n" \
               "Блестит на крапиве.\n"\
               "Я стою у дороги,\n"\
               "Прислонившись к иве.\n"\
               "От луны свет большой\n"\
               "Прямо на нашу крышу.\n"\
               "Где-то песнь соловья\n"\
               "Хорошо и тепло,\n"\
               "Как зимой у печки.\n"\
               "И березы стоят,\n"\
               "Как большие свечки.\n"\
               "И вдали за рекой,\n"\
               "Видно, за опушкой,\n"\
               "Сонный сторож стучит\n"\
               "Мертвой колотушкой."
        markup, result = MetreClassifier.improve_markup(Markup.process_text(text, self.stress_predictor))
        print(result.metre)
        self.assertTrue(result.metre == "dolnik3" or result.metre == "dolnik2")

    def test_metre_classifier5(self):
        text = "Глыбу кварца разбили молотом,\n" \
               "И, веселым огнем горя,\n" \
               "Заблестели крупинки золота\n" \
               "В свете тусклого фонаря.\n" \
               "И вокруг собрались откатчики:\n" \
               "Редкий случай, чтоб так, в руде!\n" \
               "И от ламп заплясали зайчики,\n" \
               "Отражаясь в черной воде...\n" \
               "Прислонившись к мокрой стене,\n" \
               "Мы стояли вокруг.\n" \
               "Курили,\n" \
               "Прислонившись к мокрой стене,\n" \
               "И мечтательно говорили\n" \
               "Не о золоте — о весне.\n" \
               "И о том, что скоро, наверно,\n" \
               "На заливе вспотеет лед\n" \
               "И, снега огласив сиреной,\n" \
               "Наконец придет пароход...\n" \
               "Покурили еще немного,\n" \
               "Золотинки в кисет смели\n" \
               "И опять — по своим дорогам,\n" \
               "К вагонеткам своим пошли.\n" \
               "Что нам золото? В дни тяжелые\n" \
               "Я от жадности злой не слеп.\n" \
               "Самородки большие, желтые\n" \
               "Отдавал за табак и хлеб.\n" \
               "Не о золоте были мысли...\n" \
               "В ночь таежную у костра\n" \
               "Есть над чем поразмыслить в жизни,\n" \
               "Кроме\n" \
               "Золота-серебра."
        markup, result = MetreClassifier.improve_markup(Markup.process_text(text, self.stress_predictor))
        self.assertTrue(result.metre == "dolnik3" or result.metre == "dolnik2")
