import pandas as pd

data_dict = {
    'name': [],
    'age': [],
    'email': []
}

data_list = []

data_list2 = []

for i in range(5):
    data_dict['name'].append(f'user_{i}')
    data_dict['age'].append(20 + i)
    data_dict['email'].append(f'user_{i}@gmail.com')

df = pd.DataFrame(data_dict)

print(df)

for i in range(5):
    row = {'name': f'user_{i}',
           'age': 20 + i,
           'email': f'user_{i}@gmail.com'}
    data_list.append(row)

df2 = pd.DataFrame(data_list)
print(df2)

for i in range(5):
    name = f'user_{i}'
    age = 20 + i
    email = f'user_{i}@gmail.com'
    data_list2.append([name, age, email])

df3 = pd.DataFrame(data_list2, columns = ['name', 'age', 'email'])
print(df3)