import pandas as pd

data_dic = {
    'name': [],
    'age': [],
    'email': []
}

data_list = []

# for i in range(5):
#     name = f'User_{i}'
#     age = 20 + i
#     email = f"user_{i}@gmail.com"
#     data_list.append([name, age, email])

# df = pd.DataFrame(data_list, columns = ['name', 'age', 'email'])
# print(df)

# for i in range(5):
#     row = {
#         'name': f"user_{i}",
#         'age': 20 + i,
#         'email': f"user_{i}@gmaail.com"
#     }
#     data_list.append(row)

# df = pd.DataFrame(data_list)

# print(df)

for i in range(5):
    data_dic['name'].append(f'user_{i}')
    data_dic['age'].append(20 + i)
    data_dic['email'].append(f'user_{i}@gmail.com')

df = pd.DataFrame(data_dic)
print(df)