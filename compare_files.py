import difflib
import sys

def compare_files(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
    
    diff = difflib.unified_diff(
        file1_lines, file2_lines,
        fromfile=file1, tofile=file2,
        lineterm=''
    )
    
    return '\n'.join(diff)

if __name__ == "__main__":
    file1 = "tools/train.py"
    file2 = "tools/train_for_twcc.py"
    
    result = compare_files(file1, file2)
    
    with open("diff_result.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
    print("Comparison complete. Results saved to diff_result.txt")
    
    # Also print the key differences in a more readable format
    print("\nKey differences:")
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
    
    matcher = difflib.SequenceMatcher(None, file1_lines, file2_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            print(f"\nLine {i1+1}-{i2} in {file1}:")
            for line in file1_lines[i1:i2]:
                print(f"  - {line.strip()}")
            print(f"Line {j1+1}-{j2} in {file2}:")
            for line in file2_lines[j1:j2]:
                print(f"  + {line.strip()}")