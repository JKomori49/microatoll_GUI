# 🌊 Microatoll Simulator (GUI Version)

A graphical simulator for coral microatoll growth and sea-level reconstruction.

*insert GUI snapshot here*

---

## 🧭 Installation and Launch Guide

### 🖥️ Visual Studio Code (Recommended)

#### 1. Buiild a Python environment

> **If you don’t have a Python environment yet (install from scratch):**  
> Follow the steps below to set up Visual Studio Code and Python together.

1. **Install [Visual Studio Code](https://code.visualstudio.com/)**

2. **Install [Python ≥3.10](https://www.python.org/downloads/)**  
   During installation, make sure to check the box:  
   **✅ “Add Python to PATH”**

3. **Launch VS Code and Install Git**  
   - When prompted, install the **Python Extension**
		- Press **`Ctrl + Shift + x`** to open Extensions
		- Enter *python* in the search bar and install Python and Pylance (provided by Microsoft)
   - Make sure that Git is installed
		- Open Terminal in VS code ( [View -> Terminal] or Press **```Ctrl + Shift + ` ```**)
		- Type the following command and enter
			```bash
			git --version
			```
		- If git is not found (e.g., <font color="#CD5C5C">git : The term 'git' is not recognized as the name of a cmdlet, function, script file, or operable program</font>...), run the following commands:
		
			Windows:
			```bash
			winget install --id Git.Git -e
			```
		
			macOS:
			```bash
			xcode-select --install
			```
		
			Ubuntu:
			```bash
			sudo apt install git
			```
		- Relaunch VS code once installation is completed
	- If you cannot install git via command line, go [git website](https://git-scm.com/install/) and download installer.
#### 2. Clone the Repository from GitHub
1. **Launch Visual Studio Code**
2. **Open the Command Palette**  
	- Press **`Ctrl + Shift + P`** (Windows/Linux)
3. **Search for “Clone Repository”**  
	- Type `Clone Repository` and select **“Git: Clone”** -> **“🐱Clone from GitHub”**
4. **Sign in to GitHub**  
	- If you are required to sign in to GitHub, select [Allow], sign in via browser, and [Authorize Visual-Studio-Code]
5. **Enter the repository URL**
	```
	https://github.com/JKomori49/microatoll_GUI.gui
	```
6. **Choose a local folder**  
	- Select where you want to save the project (e.g., `Documents\GitHub\`)
7. **Open the cloned folder**  
	- When prompted “Open cloned repository?”, click **Open**

#### 3. Install the Simulator Package
1. Open the integrated terminal:
	-  [View -> Terminal] or Press **```Ctrl + Shift + ` ```**

2. Run:
	```bash
	pip install -e .
	```