import os

def save_folder_structure(startpath, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

# 사용 예시
start_path = 'C:/programming/Imageset model/images_default_only'  # 데이터셋 경로
output_file = 'C:/programming/Imageset model/folder_structure.txt'  # 저장할 파일 경로
save_folder_structure(start_path, output_file)
