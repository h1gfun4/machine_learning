# Быстро получить столбцы и их типы
df = pd.read_csv('data/PATH.csv').drop('целевая переменная', axis=1)

types = {
    'int64': 'int',
    'float64': 'float'
}
for k, v in df.dtypes.iteritems():
    print(f'{k}: {types.get(str(v), "str")}')