import pandas as pd
import numpy as np

count = [64, 64, 128, 60, 64, 64, 64, 128, 64, 132, 64, 32, 32, 63, 1]
age = ['Q', 'Q', 'Z', 'L', 'L','L','Z','Q','Q','L','Q','Z','Z','L','L']
income = ['H','H','H','Z','D','D','D','Z','D','Z','Z','Z','H','Z','Z']
student = ['N','N','N','N','Y','Y','Y','N','Y','Y','Y','N','Y','N','N']
credit = ['L','Y','L','L','L','Y','Y','L','L','L','Y','Y','L','Y','Y']

result = ['N','N','Y','Y','Y','N','Y','N','Y','Y','Y','Y','Y','N','Y']

assert(len(count) == len(age) == len(income) == len(student) == len(credit) == len(result))


df = pd.DataFrame({
    'count': count,
    'age': age,
    'income': income,
    'student': student,
    'credit': credit,
    'result': result
})
print(df)

def helper(df: pd.DataFrame, dim=0):
    print_prefix = ' ' * (dim*5)
    def calc_entropy(tmp_d: pd.DataFrame):
        all_len = []
        tmp_d_attributes = tmp_d['result'].unique()
        for tmp_d_att in tmp_d_attributes:
            tmp_tmp_d = tmp_d[tmp_d['result'].values == tmp_d_att]
            tmp_tmp_num = tmp_tmp_d['count'].sum()
            all_len.append(tmp_tmp_num)
        all_len = np.array(all_len, dtype=np.float64)
        all_len /= all_len.sum()
        tmp_all_len = np.log2(all_len)
        return -(all_len*tmp_all_len).sum()

    original_entropy = calc_entropy(df)
    print(f'{print_prefix}Original Entropy={original_entropy}')
    if (original_entropy < 1e-5):
        print(f'{print_prefix}Already converge')
        return 

    attributes = ['age', 'income', 'student', 'credit']
    number = df['count'].sum()

    sub_entropy, min_sub_entropy, choose_att = original_entropy, original_entropy, None

    for att in attributes:
        sub_attributes = df[att].unique()
        sub_entropy = 0
        if len(sub_attributes) == 1:
            continue
        for sub_att in sub_attributes:
            sub_d = df[df[att].values == sub_att]
            sub_number = sub_d['count'].sum()
            g = calc_entropy(sub_d)
            sub_entropy += (sub_number / number) * g
            print(f'{print_prefix}{att}[{sub_att}] = {g}, quotio = {sub_number}//{number}')
        print(f'{print_prefix}	{att} = {sub_entropy}, gain [{original_entropy-sub_entropy}]')
        
        if sub_entropy < min_sub_entropy - 1e-5:
            min_sub_entropy = sub_entropy
            choose_att = att
    
    if choose_att != None:
        print(f'{print_prefix}Choose {choose_att}')
        all_sub_att = df[choose_att].unique()
        for k in all_sub_att:
            print(f'{print_prefix}-----Start Dim={dim}, Choose {k} ---------')
            helper(df[df[choose_att] == k], dim+1)
            print(f'{print_prefix}-----End Dim={dim}, Choose {k} ---------')
    else:
        print(f'{print_prefix}Already converge')



helper(df)