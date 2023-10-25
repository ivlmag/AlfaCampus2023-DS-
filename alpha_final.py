import csv
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

df_test = pd.read_csv('df_test.csv', sep=';')

def predict_mcc_codes(dataframe,x=0,N=10):
	'''Предсказание будущих транзакций

	Параметры:
	dataframe - объект DataFrame (тренировочный или тестовый)

	x - смещение чувствительности транзакции относительно положения в датасете Data:
		0 - без смещения (по умолчанию); 
		<0 - смещение в сторону более старых транзакций;
		>0 - смещение в сторону более новых транзакций

	N - количество требуемых предсказаний (по умолчанию 10)'''

	dataframe['Predicted']=''
	for line in range(0,len(dataframe)):
		seq = dataframe['Data'][line].split(',')

		frequency, prediction_db={},{}
		length_with_x=0

		#Этап 1: распределяем N предсказаний пропорционально частоте mcc кодов пользователя
		
		"""Подготовка словаря с частотой mcc кодов пользователя с учетом заданного смещения"""
		for i in range(0,len(seq)):
			mcc=seq[i]
			add_x=1+i*(x/(len(seq)))
			frequency[mcc]=add_x if (mcc not in frequency) else (frequency[mcc]+add_x)
			length_with_x+=add_x
		
		"""Сохраняем N предсказанных транзакций с учетом частоты mcc кодов и заданного смещения 
		Так как из-за округления транзакций может оказаться меньше N, то остаток
		заполняем самым распространенным кодом.
		Убираем пустые элементы и сортируем словарь по убыванию числа кодов"""
		prediction_db={mcc:(round(freq/length_with_x*N)) for (mcc,freq) in frequency.items()}
		top_mcc=max(set(seq), key=seq.count)
		while sum(prediction_db.values())<N:
			prediction_db[top_mcc]+=1

		prediction_db=dict((mcc,freq) for mcc,freq in prediction_db.items() if freq)
		prediction_db=dict(sorted(prediction_db.items(), key=lambda x:x[1], reverse=True))

		#Этап 2: полученные будущие транзакции равномерно распределяем в предсказании

		"""Расчет коэффициента равномерного распределения"""
		k=len(prediction_db)/N

		"""Распределяем коды из полученного на первом этапе словаря"""
		prediction=[]
		while sum(prediction_db.values())!=0:
			for mcc, n in prediction_db.items():
				times=1 if (0<n*k<1) else round(n*k)
				prediction.extend([mcc]*times)
				prediction_db[mcc]-=times

		dataframe['Predicted'][line]=','.join(prediction)

#В результате предварительного анализа (см. вложенный блокнот IzhevskiyVL.ipynb)
#лучшие результаты показал коэффициент смещения "-5", поэтому пересчитываем на него
for x in [-5]:
	predict_mcc_codes(dataframe=df_test,x=x)

	submission=df_test[['Id','Predicted']]
	submission['Predicted']=submission.Predicted.astype(str).str.replace(',',' ')
	submission['Predicted']='['+submission['Predicted']+']'
	filename = 'submission Izhevskiy X =' + str(x) + '.csv'
	submission.to_csv(filename, index=False,quoting=csv.QUOTE_NONE)