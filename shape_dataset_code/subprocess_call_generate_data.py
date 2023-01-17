import subprocess

for i in range(750):
    result = subprocess.run("python generate_data.py", stdout=subprocess.PIPE)
    print(result.stdout)