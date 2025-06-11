# GUI-point-defects
[XRDlicious](http://xrdlicious.com/) submodule for creating random point defects (interstitials, substitutes, vacancies) in crystal structures. Try it here: [xrdlicious-point-defects.streamlit.app/](https://xrdlicious-point-defects.streamlit.app/)

![Point defects module illustration](Point_Defects_Module/point_defects_1.png)

For more computationally demanding calculations with more extensive data, please compile the code locally on your computer (follow the manual below).
# **How to compile and run the XRDlicious submodule for point defects locally:** 

### **Prerequisities**: 
- Python 3.x (Tested 3.12)
- Console (For Windows, I recommend to use WSL2 (Windows Subsystem for Linux))
- Git (optional for downloading the code)
  


### **Compile the app**  
Open your terminal console and write the following commands (the bold text):  
(Optional) Install Git:  
      **sudo apt update**  
      **sudo apt install git**    
      
1) Download the XRDlicious code from GitHub (or download it manually without Git on the following link by clicking on 'Code' and 'Download ZIP', then extract the ZIP. With Git, it is automatically extracted):  
      **git clone https://github.com/bracerino/GUI-point-defects.git**

2) Navigate to the downloaded project folder:  
      **cd GUI-point-defects/**

3) Create a Python virtual environment to prevent possible conflicts between packages:  
      **python3 -m venv point_defects_env**

4) Activate the Python virtual environment (before activating, make sure you are inside the xrdlicious folder):  
      **source point_defects_env/bin/activate**
   
5) Install all the necessary Python packages:  
      **pip install -r requirements.txt**

6) Run the XRDlicious app (always before running it, make sure to activate its Python virtual environment (Step 4):  
      **streamlit run app.py**

### Workflow illustration
- Upload structure and create supercell
- Select mode and its settings (introduce interstitials, vacancies, or substitutes). For less than 500 atoms in the structure, it is possible to select to place the defects either as far away as possible, as near as possible, or something in between these two cases using fast greedy algorithm (see the description of the application website). For higher number, it is possible to place the defects only randomly do to large computational demand.
![Select point defects mode](Point_Defects_Module/point_defects_1.png)
- Apply the settings and download the modified defected structure.
![Apply settings and download structure](Point_Defects_Module/point_defects_1.png)
