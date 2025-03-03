import os

output_file = "merged_script.py"

def get_all_python_files(directory):
    """ 遞迴獲取目錄下所有 .py 檔案的完整路徑 """
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

# 取得所有 Python 檔案
python_files = get_all_python_files(".")

# 開啟輸出檔案，並將所有檔案內容寫入
with open(output_file, "w", encoding="utf-8") as outfile:
    for file_path in python_files:
        with open(file_path, encoding="utf-8") as infile:
            outfile.write(f"# --- Start of {file_path} ---\n")
            outfile.write(infile.read())
            outfile.write(f"\n# --- End of {file_path} ---\n\n")

print(f"所有 Python 檔案已合併到 {output_file}")