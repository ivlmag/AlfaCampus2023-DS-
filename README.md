# AlfaCampus2023-DS-
Решение вступительной задачи в Alfa Campus 2023 года (поток Data Science в финтехе)

# Полный текст задания:

Одним из самых ценных источников информации о клиенте являются данные о банковских транзакциях. В этом соревновании участникам предлагается предсказать будущие траты клиента, используя информацию о совершенных тратах.

# File descriptions:
- df_train.csv - данные для обучения предиктивного алгоритма.
- df_test.csv - тестовый датасет
- submission_file.csv - пример файла сабмита

# Data fields:
- df_train.csv:
  + data - история трат клиентов (последовательность mcc-кодов в хронологическом порядке); 
  + target - последовательность будущих трат клиента
- df_test.csv: 
  + Id - идентификатор клиента; 
  + data - история трат клиентов (последовательность mcc-кодов в хронологическом порядке)

# Решения
- IzhevskiyVL.ipynb - блокнот Jupyter Notebook
- alpha_final.py - чистый код
- Более подробное описание решения можно прочесть здесь: https://medium.com/@ivlmag/вступительное-задание-в-alfa-campus-2023-поток-ds-в-финтехе-bd02312b4302 
