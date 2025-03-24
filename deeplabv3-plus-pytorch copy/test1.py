import subprocess

# 1. 输入图片地址并保存
img_url = input("请输入图片地址: ")

# 2. 指定 train.py 的完整路径（或相对路径）
train_script_path = r"F:\DEEPLABV3\deeplabv3-plus-pytorch\predict.py"  # 替换为你实际的 train.py 地址

# 3. 调用 train.py，并将图片地址作为标准输入传入
process = subprocess.Popen(
    ["python", train_script_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 4. 通过 communicate 方法发送图片地址（记得添加换行符）
output, error = process.communicate(img_url + "\n")

print("train.py 输出:")
print(output)
if error:
    print("错误信息:")
    print(error)
