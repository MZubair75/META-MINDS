# 🚀 **META MINDS - Project Running Guide**

This guide provides comprehensive instructions for running the Meta Minds AI-powered data analysis platform using different methods and configurations.

---

## 📋 **Prerequisites**

Before running Meta Minds, ensure you have:

- **Python 3.13+** installed on your system
- **OpenAI API key** (required for GPT-4 integration)
- **CSV/Excel/JSON datasets** to analyze
- **Internet connection** for AI API calls

---

## 🎯 **Option 1: Standard Command Line Execution (Recommended)**

### **Step 1: Setup Environment**
```bash
# Navigate to project directory
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"

# Create .env file with your API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### **Step 2: Run the Platform**
```bash
# Standard execution with Python 3.13
py -3.13 main.py
```

### **What Happens:**
1. ✅ **System Check**: Validates Python version, API key, dependencies
2. 🎯 **Analysis Mode Selection**: Choose SMART Analysis (recommended)
3. 📝 **Context Collection**: Select business template or provide custom context
4. 🔢 **Question Configuration**: Set question counts (15 individual + 10 comparison)
5. 📁 **Dataset Input**: Provide paths to your data files
6. 🚀 **AI Processing**: Generates high-quality analytical questions
7. 📊 **Report Generation**: Creates professional reports in `/Output` folder

---

## 🎯 **Option 2: Virtual Environment Execution**

### **Step 1: Activate Virtual Environment**
```bash
# Navigate to project directory
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"

# Activate virtual environment (Windows)
venv\Scripts\activate

# For PowerShell
venv\Scripts\Activate.ps1

# For Command Prompt
venv\Scripts\activate.bat
```

### **Step 2: Run with Virtual Environment**
```bash
# After activation, run normally
python main.py
```

### **Step 3: Deactivate (After Use)**
```bash
# Deactivate virtual environment
deactivate
```

**💡 Note**: If virtual environment is corrupted, use Option 1 with system Python.

---

## 🎯 **Option 3: Direct Python Version Execution**

### **Method 3A: Python Launcher (Windows)**
```bash
# Use Python launcher to specify version
py -3.13 main.py

# Alternative version specification
py -3 main.py

# Check available Python versions
py -0
```

### **Method 3B: Full Python Path**
```bash
# If Python is in PATH
python3.13 main.py

# With full path (adjust path as needed)
"C:\Python313\python.exe" main.py
```

### **Method 3C: Alternative System Python**
```bash
# Try different Python commands
python main.py
python3 main.py
py main.py
```

---

## 🎯 **Option 4: IDE/Editor Execution**

### **Visual Studio Code**
1. Open project folder in VS Code
2. Open `main.py`
3. Select Python interpreter (3.13+)
4. Press **F5** or click **Run Python File**
5. Use integrated terminal for input

### **PyCharm**
1. Open project in PyCharm
2. Configure Python interpreter (3.13+)
3. Right-click `main.py` → **Run 'main'**
4. Use console for interactive input

### **Jupyter Notebook/Lab**
```python
# In a new notebook cell
%cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
%run main.py
```

---

## 🎯 **Option 5: Batch File Execution (Windows)**

### **Create Run Script**
Create `run_meta_minds.bat`:
```batch
@echo off
cd /d "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
py -3.13 main.py
pause
```

### **Execute Script**
```bash
# Double-click the .bat file or run from command line
run_meta_minds.bat
```

---

## 🎯 **Option 6: PowerShell Script Execution**

### **Create PowerShell Script**
Create `run_meta_minds.ps1`:
```powershell
# Set execution policy (run once as administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Navigate and run
Set-Location "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
py -3.13 main.py
```

### **Execute Script**
```powershell
# Run PowerShell script
.\run_meta_minds.ps1
```

---

## 🎯 **Option 7: Scheduled/Automated Execution**

### **Task Scheduler (Windows)**
1. Open **Task Scheduler**
2. Create **Basic Task**
3. Set **Trigger** (daily, weekly, etc.)
4. Set **Action**: Start a program
5. **Program**: `py`
6. **Arguments**: `-3.13 main.py`
7. **Start in**: Project directory path

### **Command Line Scheduling**
```bash
# Schedule daily execution at 9 AM
schtasks /create /tn "MetaMinds" /tr "py -3.13 main.py" /sc daily /st 09:00 /sd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
```

---

## 🎯 **Option 8: Network/Remote Execution**

### **Remote Desktop**
1. Connect to remote machine via RDP
2. Follow any standard execution method
3. Transfer datasets and retrieve reports

### **SSH (if applicable)**
```bash
# Connect via SSH (if SSH server enabled)
ssh user@remote-machine
cd "/path/to/META_MINDS_INDIVIDUAL"
python3.13 main.py
```

---

## 🔧 **Troubleshooting Different Execution Methods**

### **Issue 1: Python Not Found**
```bash
# Check Python installation
py --version
python --version
python3 --version

# Install Python 3.13 if missing
# Download from: https://python.org/downloads/
```

### **Issue 2: Virtual Environment Issues**
```bash
# Bypass virtual environment
py -3.13 main.py

# Recreate virtual environment
python -m venv new_venv
new_venv\Scripts\activate
pip install -r requirements.txt
```

### **Issue 3: Permission Errors**
```bash
# Run as administrator (if needed)
# Right-click PowerShell → "Run as administrator"

# Or use different user permissions
runas /user:administrator "py -3.13 main.py"
```

### **Issue 4: Path Issues**
```bash
# Use absolute paths
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"

# Check current directory
pwd   # Linux/Mac
cd    # Windows
```

### **Issue 5: API Key Issues**
```bash
# Check .env file exists
dir .env
ls .env

# Verify API key format
type .env
cat .env

# Recreate .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## 📊 **Execution Method Comparison**

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Standard CLI** | Simple, reliable, system Python | None | Most users |
| **Virtual Environment** | Isolated dependencies | Setup complexity, can break | Development |
| **Python Launcher** | Version control, flexible | Windows only | Multiple Python versions |
| **IDE Execution** | Debugging, integrated tools | IDE overhead | Development/debugging |
| **Batch Script** | One-click execution, automation | Windows only | Regular users |
| **PowerShell** | Advanced scripting, automation | Permission issues | Power users |
| **Scheduled** | Automated execution | Setup complexity | Regular reports |
| **Remote** | Server execution, scaling | Network dependency | Enterprise use |

---

## 🎯 **Quick Start Commands by System**

### **Windows (Recommended)**
```bash
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
py -3.13 main.py
```

### **Windows (Alternative)**
```bash
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
python main.py
```

### **Windows (Virtual Environment)**
```bash
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"
venv\Scripts\activate
python main.py
```

### **Linux/Mac (If Applicable)**
```bash
cd "/path/to/META_MINDS_INDIVIDUAL"
python3.13 main.py
```

---

## 📋 **Pre-Execution Checklist**

Before running Meta Minds, verify:

- [ ] **Python 3.13+** installed and accessible
- [ ] **OpenAI API key** in `.env` file
- [ ] **Project directory** is correct
- [ ] **Datasets ready** (CSV/Excel/JSON files)
- [ ] **Internet connection** active
- [ ] **Terminal/Command Prompt** open in project directory

---

## 🎯 **Expected Output Structure**

After successful execution, you'll find:

```
Output/
├── Individual_[Focus]_[Objective]_[Audience]_[DateTime].txt
└── Cross-Dataset_[Focus]_[Objective]_[Audience]_[DateTime].txt

user_context.json (updated with preferences)
```

**Example Files:**
- `Individual_Salesperformance_Riskassessment_Executives_2025-01-08_14-30.txt`
- `Cross-Dataset_Salesperformance_Riskassessment_Executives_2025-01-08_14-30.txt`

---

## 💡 **Best Practices**

1. **Use Option 1** (Standard CLI) for most reliable execution
2. **Keep datasets organized** in accessible folders
3. **Use consistent API keys** across sessions
4. **Check Output folder** for generated reports
5. **Review user_context.json** for saved preferences
6. **Use meaningful dataset names** for easier identification
7. **Ensure sufficient disk space** for report generation

---

## 🆘 **Emergency Execution (If Nothing Works)**

If all methods fail, try this minimal approach:

```bash
# 1. Download fresh Python 3.13
# 2. Install manually from python.org
# 3. Open Command Prompt as Administrator
# 4. Navigate to project folder
cd "C:\Users\Jatin\Documents\Automations - DS\META_MINDS_INDIVIDUAL"

# 5. Install dependencies manually
pip install openai crewai pandas langchain-openai

# 6. Create API key file manually
echo OPENAI_API_KEY=your_key_here > .env

# 7. Run with full path
C:\Python313\python.exe main.py
```

---

**Choose the execution method that best fits your environment and expertise level. The Standard CLI method (Option 1) is recommended for most users.** 🚀
