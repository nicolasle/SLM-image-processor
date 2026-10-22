import os
import subprocess

def create_shortcuts():
    work_dir = os.path.abspath(os.path.dirname(__file__))
    python_exe = os.path.join(os.path.dirname(subprocess.__file__), "..", "pythonw.exe")
    python_exe = os.path.abspath(python_exe)
    if not os.path.exists(python_exe):
        import sys
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    
    script_path = os.path.join(work_dir, "main.py")
    icon_path = os.path.join(work_dir, "app_icon.ico")
    desktop_dir = os.path.join(os.environ["USERPROFILE"], "Desktop")

    ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell

$sc1 = $WshShell.CreateShortcut('{os.path.join(work_dir, "SLM Image Processor.lnk")}')
$sc1.TargetPath = '{python_exe}'
$sc1.Arguments = '"{script_path}"'
$sc1.WorkingDirectory = '{work_dir}'
$sc1.IconLocation = '{icon_path}'
$sc1.Description = 'Scanning Laser Microscopy Image Processor'
$sc1.Save()

if (Test-Path '{desktop_dir}') {{
    $sc2 = $WshShell.CreateShortcut('{os.path.join(desktop_dir, "SLM Image Processor.lnk")}')
    $sc2.TargetPath = '{python_exe}'
    $sc2.Arguments = '"{script_path}"'
    $sc2.WorkingDirectory = '{work_dir}'
    $sc2.IconLocation = '{icon_path}'
    $sc2.Description = 'Scanning Laser Microscopy Image Processor'
    $sc2.Save()
}}
"""
    ps_file = os.path.join(work_dir, "_create_sc.ps1")
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write(ps_script)

    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_file], check=True)
    if os.path.exists(ps_file):
        os.remove(ps_file)
    print("Shortcuts created successfully.")

if __name__ == "__main__":
    create_shortcuts()
