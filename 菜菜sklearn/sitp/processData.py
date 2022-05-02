import pandas as pd

# 读取数据
bank_data = pd.read_csv('.\\interbank_data\\interbank_2020.csv')
ranking_data = pd.read_csv('.\\ranking\\ranking_2020.csv')

# 数据合并
all_data_st = pd.merge(bank_data, ranking_data, how='left', on=['id'])
all_data_st.dropna(axis=0, how='any', inplace=True, subset=None)

# 导出结果数据
all_data_st.to_csv(".\\data\\2020.csv")
