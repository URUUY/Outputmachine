import os
from PIL import Image

def convert_jpg_to_png(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            # 构建完整的文件路径
            jpg_path = os.path.join(folder_path, filename)
            
            # 构建新的png文件名和路径
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(folder_path, png_filename)
            
            try:
                # 打开JPG图像并保存为PNG
                with Image.open(jpg_path) as img:
                    img.save(png_path, 'PNG')
                print(f"转换成功: {filename} -> {png_filename}")
                
                # 可选: 删除原始的JPG文件
                # os.remove(jpg_path)
                
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ")
    if os.path.isdir(folder_path):
        convert_jpg_to_png(folder_path)
    else:
        print("无效的文件夹路径")