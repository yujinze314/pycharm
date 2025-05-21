import os

# 路径修改为你的实际路径
base_dir = r"E:/gated2depth/Gated2Depth-master/data/real"
txt_path = r"E:/gated2depth/Gated2Depth-master/splits/real_test_night.txt"

# 读取所有ID
with open(txt_path, 'r') as f:
    ids = [line.strip() for line in f if line.strip()]

missing = []

for img_id in ids:
    for gate_id in range(3):
        img_path = os.path.join(base_dir, f"gated{gate_id}_10bit", f"{img_id}.png")
        if not os.path.isfile(img_path):
            missing.append(img_path)

if not missing:
    print("所有图片都存在！")
else:
    print(f"缺失图片数量: {len(missing)}")
    for path in missing:
        print("缺失：", path)
