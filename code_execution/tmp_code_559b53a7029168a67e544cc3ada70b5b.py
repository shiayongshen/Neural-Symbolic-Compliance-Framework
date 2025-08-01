import os
import subprocess
                                                                                                           
def main():
    result = subprocess.run(
        ['python', 'case_case_0_code.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'  
    )
    return result.stdout

if __name__ == "__main__":
    output = main()
    print(output)