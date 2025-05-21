import pandas as pd
import re

results = []
with open('results_only.txt', 'r', encoding='utf-8') as f:
    for line in f:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        # 只取第1、3、5、7、9、11个（下标0、2、4、6、8、10）
        if len(nums) >= 11:
            needed = [float(nums[i]) for i in [0,2,4,6,8,10]]
            results.append(needed)
        # 如果不足11个数字，可选择打印出来排查
        elif len(nums) > 0:
            print("行格式异常:", line.strip(), nums)

df = pd.DataFrame(results, columns=['rmse', 'ard', 'mae', 'delta1', 'delta2', 'delta3'])
print(df.head(10))
df.to_csv("results_only_cleaned.csv", index=False)