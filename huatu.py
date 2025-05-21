import pandas as pd
import matplotlib.pyplot as plt

# 请确保这里读取的是你刚才清洗过的数据文件
df = pd.read_csv("results_only_cleaned.csv")

print(df.head(10))

plt.figure(figsize=(10,6))
plt.plot(df['rmse'], label='RMSE')
plt.plot(df['mae'], label='MAE')
plt.plot(df['ard'], label='ARD')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Evaluation Metrics per Sample')
plt.legend()
plt.show()