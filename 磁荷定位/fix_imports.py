import os
import re

def fix_relative_imports(directory):
    pattern = r'from \.\.([\w\.]+) import'
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找并替换相对导入
                modified_content = re.sub(pattern, r'from \1 import', content)
                
                if content != modified_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    print(f"Fixed imports in {filepath}")

if __name__ == "__main__":
    # 替换为您的项目根目录
    project_dir = "."
    fix_relative_imports(project_dir)