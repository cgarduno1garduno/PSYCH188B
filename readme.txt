PSYCH 188B
Final Project   Dataset: fMRI
----------------------------------------------------------------------------------------------------

   GROUP MEMBERS
--------------------
Bryce Wong
Cristopher Garduno    
Haesoo Kim
Pratyusha Javangula


SETUP
----------------------------------------------------------------------------------------------------
1) Ensure that your test files are inside a folder named '/haxby2001-188B'
    If you want to change the name of the folder, edit line 306 (below) in run_all_2-7.py, and change 
    /haxby2001-188B to your desired file name. 
        cwd = os.getcwd()+'/haxby2001-188B'

2) Place the test models and encoder in the same directory as your test files. 

3) Open the terminal/command prompt and navigate to the parent directory of the folder containing
    your test files. 

4) Run the run_all function (see USAGE)

USAGE
----------------------------------------------------------------------------------------------------
run_all_2-7.py:

  In the python shell, run the following command to execute the file and define all of the necessary functions.
  
        >>> execfile('run_all_2-7.py')
        
  Now, enter the following command to test your data. 
  
        >>> run_all()
