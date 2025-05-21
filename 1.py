import os

def validate_dataset(base_dir, filenames):
    missing_files = []
    for img_id in filenames:
        for gate_id in range(3):
            gate_dir = os.path.join(base_dir, f'gated{gate_id}_10bit')
            path = os.path.join(gate_dir, f'{img_id}.png')
            if not os.path.exists(path):
                missing_files.append(path)
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)
    else:
        print("All files are present.")

# 调用验证函数
base_dir = 'E:/gated2depth/Gated2Depth-master/data/Real/'
filenames = ["10842", "10843"]  # 示例文件名列表
validate_dataset(base_dir, filenames)
