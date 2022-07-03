from fileinput import filename
from pydoc import ModuleScanner
from unittest import result
import matplotlib.pyplot as plt
from ansys.mapdl import reader as pymapdl_reader
from ansys.mapdl.core import launch_mapdl
import ansys.dpf.core as dpf
import numpy as np
import pathlib
import pickle as pkl
import extract_result_ansys as ERA
import os
import shutil
from tqdm import tqdm
import pprint as pp

MASS_ELEMENT = "mass_element"
SPR_ELEMENT = "spring_element"
AC_ELEMENT = "acoustic_element"
AC_ELEMENT_3D = "acoustic_element_3D"
AC_LIM_ELEMENT = "acoustic_limit_element"
MPC_ELEMENT = "mpc_element"
STR_ELEMENT = "structural_element"
STR_ELEMENT_3D = "structural_element_3D"
DENS = "density"
OMEG = "fluid_omega"
SOUND_VEL = "celerity"
STRUCTURAL_SOUND_VEL= "structural_sound_celerity"
E_MODULUS = "young-modulus"
POISS = "poisson-coefficient"
PARAM_FOLDER = pathlib.Path(os.getcwd()) / "parametric-sim"

class Fe_model:
    def __init__(self, model_name:str, db_name:str, template_file_path:pathlib.Path, variables:dict):
        """
        Notes : Fe_model class stores all the information about the finite element model to launch
                in Ansys APDL.

        Attributes:
            model_name( type:str ): 
                The name of the model

            db_name( str ): 
                The name of the database.

            template_file_path( pathlib.Path ): 
                The location of the template file.
            
            variables( type:dict ):
                All the variables of the model
        """
        self.db_name = db_name
        self.model_name = model_name
        self.template_file_path = template_file_path
        self.variables = variables

    def write_mapdl_file(self, analysis_dir='./mapdl', verbose = True):
            """Writes the APDL commands file related to the current model object. Template file (defined with model_type)
            is read line by line and when a specific target is reached, the appropriate variables are written to the output file.

            Parameters
            ----------
            analysis_dir (str): analysis folder path, where the output file is written
            model_type (str): template model type name; the code will then read the appropriate file in the folder defined by
                apdl_model_templates_dir at the top of this Python code

            Notes
            -----
            By default, all template files don't have any fluid velocity and therefore use the unsymmetric solver
            (MODOPT,UNSYM,...). When rotation is toggled (rotation_mean_flow_effect = True), the solver is automatically changed
            to damped (MODOPT,DAMP,...) and the appropriate lines required to apply velocity at fluid nodes are added.
            """
            template_file_path = str(self.template_file_path)
            # obtain template file's full path
            if verbose:
                print("Reading template file:",template_file_path)

            # obtain template file's lines and store them in a list
            #template_file = open(template_file_path,'r')
            template_file=open(template_file_path,'r')
            template_lines = template_file.readlines()
            template_file.close()

            # obtain model file's full path
            model_file_path = ''.join((analysis_dir,'/', self.model_name))
            if verbose:
                print("Writing file:", model_file_path)

    def model_names(self):
        """Returns full model names (with extension) for APDL command file (.mac), Ansys geometry & loads database file (.dat),
        and structural results file (.rst)

        Returns
        -------
        apdl_commands_name (str): commands file name
        database_name (str): geometry & loads database file name
        result_file_name (str): structural results file name

        Notes
        -----
        Names are based on the model name defined by the user (self.db_name), the extension pertaining to each file is then added automatically.
        """

        apdl_commands_name = ''.join((self.db_name,'.mac'))
        database_name = ''.join((self.db_name,'.dat'))
        result_file_name = ''.join((self.db_name,'.rst'))

        return (apdl_commands_name,database_name,result_file_name)

    def __str__(self):        
        return vars(self)


class Disc_model(Fe_model):
    def __init__(self, model_name, db_name, template_file_path, variables, fluid_element, structural_element, fluid_density,sound_celerity, structural_density, young_modulus, poisson,result_propreties, real_constants, mass_element, spring_element,  fluid_infinite_element,
                structural_sound_celerity, dimension_3D = False, mean_flow = False, CSV_file = False, fluid_element_3D = None, structural_element_3D = None, csv_file_path = None,  flow_type = None, omega = None):
        super().__init__(model_name, db_name, template_file_path, variables)

        

        self.real_constants=real_constants
        self.mean_flow = mean_flow
        self.flow_type = flow_type
        self.csv_file_path = csv_file_path
        self.dimension_3D = dimension_3D
        self.CSV_file = CSV_file
        self.omega = omega

        elements = {}
        elements[MASS_ELEMENT]= mass_element
        elements[SPR_ELEMENT]= spring_element
        elements[AC_ELEMENT] = fluid_element
        elements[AC_ELEMENT_3D] = fluid_element_3D
        elements[AC_LIM_ELEMENT]= fluid_infinite_element
        elements[STR_ELEMENT] = structural_element
        elements[STR_ELEMENT_3D] = structural_element_3D
        self.elements = elements

        fluid_propreties = {}
        fluid_propreties[OMEG] = omega
        fluid_propreties[DENS] = fluid_density
        fluid_propreties[SOUND_VEL] = sound_celerity
        self.fluid_propreties = fluid_propreties

        structural_propreties = {}
        structural_propreties[DENS] = structural_density
        structural_propreties[STRUCTURAL_SOUND_VEL]=structural_sound_celerity #METTRE EN MODE +FALSE DANS LE INIT ??
        structural_propreties[E_MODULUS] = young_modulus
        structural_propreties[POISS] = poisson
        self.structural_propreties = structural_propreties

        self.result_propreties = result_propreties

        # MAPDL commands block added to files when mean flow is turned on (fe_model().rotation_mean_flow_effect = True)
        self.generic_mean_flow_lines_Blais = ['CSYS,12 ! activate cylindrical coordinate system\n','\n','! interpolation of mean velocity\n',
        '*DIM,val,TABLE,3,,,X,,,12 !X = Radius in cylindrical coord system\n',
        '*TAXIS,val(1),1,0.0,inner_radius,outer_water_radius ! center point, inner radius, outer radius (m)\n',
        'val(1) = 0.0,(1-K)*inner_radius*omega*3.14159265*2,(1-K)*outer_water_radius*omega*3.14159265*2\n','\n','*IF,omega,GT,0,then\n'
        '! mean velocity set to (0,r*omega_f,0)\n','BF,fluid_nodes,VMEN,0.0,%val%,0.0 !interpolate tangential component\n','*endif\n','\n',
        '! reselection of all elements and nodes\n','ESEL,all\n','NSEL,all\n','CSYS,0\n','!End of mean flow commands\n','\n']



        self.generic_mean_flow_lines_TC_flow = ['CSYS,12 ! activate cylindrical coordinate system\n','\n','! interpolation of mean velocity\n',
        '*DIM,val,TABLE,3,,,X,,,12 !X = Radius in cylindrical coord system\n',
        '*TAXIS,val(1),1,inner_radius,(inner_radius+outer_radius)/2,outer_radius ! center point, inner radius, outer radius (m)\n',
        'mid_radius = (outer_radius+inner_radius)/2 \n','val_1 = omega*inner_radius*inner_radius/(outer_radius*outer_radius-inner_radius*inner_radius)*(outer_radius*outer_radius/inner_radius-inner_radius)\n',
        'val_2 = omega*inner_radius*inner_radius/(outer_radius*outer_radius-inner_radius*inner_radius)*(outer_radius*outer_radius/mid_radius-mid_radius)\n',
        'val_3 = omega*inner_radius*inner_radius/(outer_radius*outer_radius-inner_radius*inner_radius)*(outer_radius*outer_radius/outer_radius-outer_radius)\n',
        'val(1) = val_1,val_2,val_3\n','\n','*IF,omega,GT,0,then\n',
        '! mean velocity set to (0,r*omega_f,0)\n','BF,fluid_nodes,VMEN,0.0,%val%,0.0 !interpolate tangential component\n','*endif\n','\n',
        '! reselection of all elements and nodes\n','ESEL,all\n','NSEL,all\n','CSYS,0\n','!End of mean flow commands\n','\n']


        self.generic_mean_flow_lines_cst_w = ['CSYS,12 ! activate cylindrical coordinate system\n','\n','! interpolation of mean velocity\n',
        '*DIM,val,TABLE,3,,,X,,,12 !X = Radius in cylindrical coord system\n',
        '*TAXIS,val(1),1,inner_radius,(inner_radius+outer_radius)/2,outer_radius ! center point, inner radius, outer radius (m)\n',
        'val(1) = inner_radius*omega,(inner_radius+outer_radius)/2*omega,outer_radius*omega\n','\n','*IF,omega,GT,0,then\n'
        '! mean velocity set to (0,r*omega_f,0)\n','BF,fluid_nodes,VMEN,0.0,%val%,0.0 !interpolate tangential component\n','*endif\n','\n',
        '! reselection of all elements and nodes\n','ESEL,all\n','NSEL,all\n','CSYS,0\n','!End of mean flow commands\n','\n']

        self.generic_csv_mean_flow_lines = ['CSYS,12 ! activate cylindrical coordinate system\n','\n','! interpolation of mean velocity\n',
        '\n','to_skip = 1 ! enter number of lines to skip ie 1 in this example \n', 'numlines = 4 !/INQUIRE,numlines,LINES,file_name,csv\n',
        'to_read=numlines-to_skip \n','\n', '*DEL,val,,NOPR \n', '*DIM,val,TABLE,to_read,,,X,,,12 ! table array to hold data \n',
        '*TREAD,val,file_name,csv,folder_name,to_skip \n','\n','*IF,omega,GT,0,then\n','BF,fluid_nodes,VMEN,0.0,%val%,0.0 !interpolate tangential component\n',
        '*endif\n','\n', '! reselection of all elements and nodes\n','ESEL,all\n','NSEL,all\n','CSYS,0\n','!End of mean flow commands\n','\n']


    def write_mean_flow_csv_file(self):
        the_omega = self.fluid_propreties[OMEG]
        csv_file_path = self.csv_file_path
        variables = self.variables
        for variable_name, variable_value in variables.items():
            #print (variable_name, variable_value)
            if variable_name == "inner_radius":
                R1 = variable_value
            elif variable_name == "outer_radius":
                R2 = variable_value
        flow_type = self.flow_type
        #cas de omega constant
        if flow_type == "cst_w":
            omega_string = str(round(the_omega,5))
            omega_number = the_omega
            csv_file_name = (csv_file_path / ("w_" + omega_string + ".csv"))
            csv_file = open(csv_file_name,"w")
            csv_file.write("radius, u_theta \n")
            #delta_r = 0.001
            delta_r = (R2-R1)/2
            r = R1
            while(r <= R2):
                u_theta = r*omega_number
                u_theta = round(u_theta,3)
                csv_file.write(str(r) + "," + str(u_theta) + "\n") 
                r+=delta_r
            csv_file.close()
        #ecoulement de taylor couette
        elif flow_type == "TC_flow":
            omega_number = the_omega
            omega_string = str(round(the_omega,5))
            csv_file_name = (csv_file_path / ("w_" + omega_string + ".csv"))
            csv_file = open(csv_file_name,"w")
            csv_file.write("radius, u_theta \n")
            csv_file.write("0.0,0.0 \n")
            r = R1
            delta_r = (R1+R2)/2
            while (r <= R2):
                u_theta = omega_number/(R2**2-R1**2)*(R2**2/r-r)
                u_theta = round(u_theta,1)
                csv_file.write(str(r) + "," + str(u_theta) + "\n") 
                r+=delta_r
            csv_file.close()





    def write_mapdl_file(self, analysis_dir='./mapdl', verbose = True):
            """Writes the APDL commands file related to the current model object. Template file (defined with model_type)
            is read line by line and when a specific target is reached, the appropriate variables are written to the output file.
            Parameters
            ----------
            analysis_dir (str): analysis folder path, where the output file is written
            model_type (str): template model type name; the code will then read the appropriate file in the folder defined by
                apdl_model_templates_dir at the top of this Python code
            Notes
            -----
            By default, all template files don't have any fluid velocity and therefore use the unsymmetric solver
            (MODOPT,UNSYM,...). When rotation is toggled (rotation_mean_flow_effect = True), the solver is automatically changed
            to damped (MODOPT,DAMP,...) and the appropriate lines required to apply velocity at fluid nodes are added.
            """
            
            template_file_path = self.template_file_path
            # obtain template file's full path
            if verbose:
                print("Reading template file:",template_file_path)

            # obtain template file's lines and store them in a list
            template_file = open(template_file_path,'r')
            template_lines = template_file.readlines()
            template_file.close()

            #obtain csv file name to write it later in the template
            if self.mean_flow == True and self.CSV_file == True:
                omega_string = str(round(self.omega,5))
                csv_file_path = self.csv_file_path
                csv_file_name = (csv_file_path / ("csv_file_" + omega_string + ".csv"))

            # obtain model file's full path
            model_file_path = ''.join((analysis_dir,'/', self.model_name))
            if verbose:
                print("Writing file:", model_file_path)

            # open model file for writing
            with open(model_file_path,'w') as file:
                # iterate on template file line by line
                for index,line in enumerate(template_lines):
                    # title section
                    if '/filname' in line.split(',')[0]:
                        file.write(''.join(('/filname,',self.db_name,',1','\n')))
                        continue # go to next line
                    # user input variables
                    elif '!input variables' in line:
                        file.write(line)
                        variables = self.variables
                        for variable_name,variable_value in variables.items():
                            # write all variables that are not None, and are not related to element type, rotation control, and material properties
                            if variable_value is not None and variable_name.split('_')[0] not in ['elem','rotation','MP']:
                                # write variables in '' (string in APDL)
                                if variable_name == 'database':
                                    file.write(''.join((variable_name, '=', "'", str(variable_value), "'", '\n')))
                                # write variables as is
                                else:
                                    file.write(''.join((variable_name,'=',str(variable_value),'\n')))
                        continue # go to next line
                    elif '!real constants' in line:
                        file.write(line)
                        real_constants = self.real_constants
                        r=0
                        for real_constant_name,real_constant_value in real_constants.items() :
                            r+=1
                            # write all real constants that are not None, and are not related to element type, rotation control, and material properties
                            if real_constant_value is not None and real_constant_name != 'real_constant_2' and real_constant_name != 'real_constant_4' and real_constant_name.split('_')[0] not in ['elem','rotation','MP']:
                                file.write(''.join(('r,', str(r),',',str(real_constant_value), '\n')))
                            elif real_constant_name == 'real_constant_2':
                                file.write(''.join(('r,', str(r),',',str(real_constant_value),',',str(real_constant_value),',',str(real_constant_value), '\n')))
                            elif real_constant_name == 'real_constant_4':
                                file.write(''.join(('r,', str(r),',',str(real_constant_value),',0,0', '\n')))
                        continue # go to next line
                    # fluid element type
                    elif '!fluid type' in line:
                        file.write(line)
                        if self.dimension_3D:
                            file.write(''.join(('ET,1,', self.elements[AC_ELEMENT_3D], '\n')))
                        else:
                            file.write(''.join(('ET,1,', self.elements[AC_ELEMENT], '\n')))
                        continue # go to next line
                    # structural element type
                    elif '!solid type' in line:
                        file.write(line)
                        if self.dimension_3D:
                            file.write(''.join(('ET,2,', self.elements[STR_ELEMENT_3D], '\n')))
                        else: 
                            file.write(''.join(('ET,2,', self.elements[STR_ELEMENT], '\n')))
                        continue # go to next line
                    # mass element type
                    elif '!mass type' in line:
                        file.write(line)
                        file.write(''.join(('ET,3,', self.elements[MASS_ELEMENT], '\n')))
                        continue # go to next line
                    # spring element type
                    elif '!spring type' in line:
                        file.write(line)
                        file.write(''.join(('ET,4,', self.elements[SPR_ELEMENT], '\n')))
                        file.write('!* \n')
                        file.write('KEYOPT,4,1,0 \n')
                        file.write('KEYOPT,4,3,2 \n')
                        file.write('!* \n')
                        continue # go to next line
                    # fluid infinite element type, only in 2D
                    elif '!fluid infinite type' in line:
                        file.write(line)
                        file.write(''.join(('ET,5,', self.elements[AC_LIM_ELEMENT], '\n')))
                        continue # go to next line

                    # fluid properties
                    elif '!fluid properties' in line:
                        file.write(line)
                        file.write(''.join(('MP,DENS,1,', str(self.fluid_propreties[DENS]), ',','! kg m^-3','\n')))
                        file.write(''.join(('MP,SONC,1,',str(self.fluid_propreties[SOUND_VEL]), ',','! m s^-1', '\n')))
                        continue # go to next line
                    # structural properties
                    elif '!solid properties' in line:
                        file.write(line)
                        file.write(''.join(('MP,EX,2,', str(self.structural_propreties[E_MODULUS]), ',', '! Pa', '\n')))
                        file.write(''.join(('MP,SONC,2,',str(self.structural_propreties[STRUCTURAL_SOUND_VEL]), ',', '! m s^-1', '\n')))
                        file.write(''.join(('MP,NUXY,2,', str(self.structural_propreties[POISS]), ',', '\n')))
                        file.write(''.join(('MP,DENS,2,', str(self.structural_propreties[DENS]), ',', '! kg m^-3', '\n')))
                        continue # go to next line

                    # #CSV file name to read
                    elif '!file_path' in line and self.mean_flow == True and self.csv_file_path != None:
                        flow_type = self.flow_type
                        csv_file_name = "w_" + omega_string
                        csv_folder_name = "../csv_file/" + flow_type + "/"
                        file.write(line)
                        file.write("".join(("file_name = '", str(csv_file_name),"'", "\n")))
                        file.write("".join(("folder_name = '", str(csv_folder_name),"'", "\n")))
                        continue # go to next line

                    # activate mean flow command for omega constant
                    elif line.split(',')[0].lower() == 'modopt' and (self.mean_flow == True) and (self.flow_type == "cst_w") and (self.CSV_file == False):
                        if self.fluid_propreties[OMEG] <= 10:
                            file.write(''.join(('modopt, DAMP,',str(6),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        else:
                            file.write(''.join(('modopt, DAMP,',str(16),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        file.write(''.join(('omega','=',str(self.fluid_propreties[OMEG]), '\n')))
                        file.writelines(self.generic_mean_flow_lines_cst_w) # see top of this Python script for lines block
                        continue # go to next line

                    # activate mean flow command for Couette flow
                    elif line.split(',')[0].lower() == 'modopt' and (self.mean_flow == True) and (self.flow_type == "TC_flow") and (self.CSV_file == False):
                        if self.fluid_propreties[OMEG] <= 10:
                            file.write(''.join(('modopt, DAMP,',str(6),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        else:
                            file.write(''.join(('modopt, DAMP,',str(16),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        file.write(''.join(('omega','=',str(self.fluid_propreties[OMEG]), '\n')))
                        file.writelines(self.generic_mean_flow_lines_TC_flow) # see top of this Python script for lines block
                        continue # go to next line

                    #activate mean flow command with use of a CSV file
                    elif line.split(',')[0].lower() == 'modopt' and (self.mean_flow == True) and (self.CSV_file == True):
                        if self.fluid_propreties[OMEG] <= 10:
                            file.write(''.join(('modopt, DAMP,',str(6),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        else:
                            file.write(''.join(('modopt, DAMP,',str(16),',',str(1.0),',',str(1200.),',','ON,ON', '\n')))
                        file.write(''.join(('omega','=',str(self.fluid_propreties[OMEG]), '\n')))
                        file.writelines(self.generic_csv_mean_flow_lines) # see top of this Python script for lines block
                        continue # go to next line


                    # user input variable, if user forgot to specify a variable that's important for the requested template file,
                    # the template file line is written, else it is skipped
                    elif line.split("=")[0].strip() in vars(self).keys() and len(line.split("=")) > 1:
                        if vars(self)[line.split("=")[0].strip()] is None:
                            file.write(line)
                        else:
                            continue # go to next line
                    # element type or material property line, which have already been written --> skip them
                    elif line.split(',')[0] == 'ET' or line.split(',')[0] == 'MP':
                        continue # go to next line
                    # generic line (no variables), write it to model file
                    else:
                        file.write(line)
            if verbose:
                print("APDL commands file written")

class Glass_model(Fe_model):
    def __init__(self, model_name, db_name, template_file_path, variables, fluid_element, structural_element,
                fluid_density, sound_celerity, structural_density, young_modulus, poisson, 
                result_propreties):
        self.db_name = db_name
        self.model_name = model_name
        self.template_file_path = template_file_path
        self.variables = variables

        elements = {}
        elements[AC_ELEMENT] = fluid_element
        elements[STR_ELEMENT] = structural_element
        self.elements = elements

        fluid_propreties = {}
        fluid_propreties[DENS] = fluid_density
        fluid_propreties[SOUND_VEL] = sound_celerity
        self.fluid_propreties = fluid_propreties

        structural_propreties = {}
        structural_propreties[DENS] = structural_density
        structural_propreties[E_MODULUS] = young_modulus
        structural_propreties[POISS] = poisson
        self.structural_propreties = structural_propreties

        self.result_propreties = result_propreties


    def write_mapdl_file(self, analysis_dir='./mapdl', verbose = True):
            """Writes the APDL commands file related to the current model object. Template file (defined with model_type)
            is read line by line and when a specific target is reached, the appropriate variables are written to the output file.

            Parameters
            ----------
            analysis_dir (str): analysis folder path, where the output file is written
            model_type (str): template model type name; the code will then read the appropriate file in the folder defined by
                apdl_model_templates_dir at the top of this Python code

            Notes
            -----
            By default, all template files don't have any fluid velocity and therefore use the unsymmetric solver
            (MODOPT,UNSYM,...). When rotation is toggled (rotation_mean_flow_effect = True), the solver is automatically changed
            to damped (MODOPT,DAMP,...) and the appropriate lines required to apply velocity at fluid nodes are added.
            """
            template_file_path = self.template_file_path
            # obtain template file's full path
            if verbose:
                print("Reading template file:",template_file_path)

            # obtain template file's lines and store them in a list
            template_file = open(template_file_path,'r')
            template_lines = template_file.readlines()
            template_file.close()

            # obtain model file's full path
            model_file_path = ''.join((analysis_dir,'/', self.model_name))
            if verbose:
                print("Writing file:", model_file_path)

            # open model file for writing
            with open(model_file_path,'w') as file:
                # iterate on template file line by line
                for index,line in enumerate(template_lines):
                    # title section
                    if '/filname' in line.split(',')[0]:
                        file.write(''.join(('/filname,',self.db_name,',1','\n')))
                        continue # go to next line
                    # user input variables
                    elif '!input variables' in line:
                        file.write(line)
                        variables = self.variables
                        for variable_name,variable_value in variables.items():
                            # write all variables that are not None, and are not related to element type, rotation control, and material properties
                            if variable_value is not None and variable_name.split('_')[0] not in ['elem','rotation','MP']:
                                # write variables in '' (string in APDL)
                                if variable_name == 'database':
                                    file.write(''.join((variable_name, '=', "'", str(variable_value), "'", '\n')))
                                # write variables as is
                                else:
                                    file.write(''.join((variable_name,'=',str(variable_value),'\n')))
                        continue # go to next line

                    # fluid element type
                    elif '!fluid type' in line:
                        file.write(line)
                        file.write(''.join(('ET,1,', self.elements[AC_ELEMENT], '\n')))
                        continue # go to next line
                    # structural element type
                    elif '!solid type' in line:
                        file.write(line)
                        file.write(''.join(('ET,2,', self.elements[STR_ELEMENT], '\n')))
                        continue # go to next line
                    # fluid properties
                    elif '!fluid properties' in line:
                        file.write(line)
                        file.write(''.join(('MP,DENS,1,', str(self.fluid_propreties[DENS]), ',','! kg m^-3','\n')))
                        file.write(''.join(('MP,SONC,1,',str(self.fluid_propreties[SOUND_VEL]), ',','! m s^-1', '\n')))
                        continue # go to next line
                    # structural properties
                    elif '!solid properties' in line:
                        file.write(line)
                        file.write(''.join(('MP,EX,2,', str(self.structural_propreties[E_MODULUS]), ',', '! Pa', '\n')))
                        file.write(''.join(('MP,NUXY,2,', str(self.structural_propreties[POISS]), ',', '\n')))
                        file.write(''.join(('MP,DENS,2,', str(self.structural_propreties[DENS]), ',', '! kg m^-3', '\n')))
                        continue # go to next line

                    # user input variable, if user forgot to specify a variable that's important for the requested template file,
                    # the template file line is written, else it is skipped
                    elif line.split("=")[0].strip() in vars(self).keys() and len(line.split("=")) > 1:
                        if vars(self)[line.split("=")[0].strip()] is None:
                            file.write(line)
                        else:
                            continue # go to next line
                    # element type or material property line, which have already been written --> skip them
                    elif line.split(',')[0] == 'ET' or line.split(',')[0] == 'MP':
                        continue # go to next line
                    # generic line (no variables), write it to model file
                    else:
                        file.write(line)
            if verbose:
                print("APDL commands file written")

        
def clean_up_directory(mapdl,analysis_path):
    """Removes all files that don't end in .rst, .dat, .mac from the analysis directory

    Parameters
    ----------
    mapdl (ansys.mapdl.core launch_apdl): Ansys Mechanical APDL instance
    analysis_path (str): full path of the folder where the parametric study is taking place

    Notes
    -----
    - When solving a model, APDL creates multiple files for each processor used. Example if the model is rotor.mac,
    then at the end there might be rotor.rst, rotor0.rst, rotor1.rst if two processors are used for the solution.
    The code will only keep rotor.rst since it contains all the results, and delete rotor0.rst, rotor1.rst.
    """

    print('\n',"Cleaning up analysis directory. Only .dat, .rst, .mac files are kept.")
    # iterate over all files
    for full_file_name in mapdl.list_files():
        # if file name is just name.extension
        if len(full_file_name.split('.')) == 2:
            file_name = full_file_name.split('.')[0]
            file_extension = full_file_name.split('.')[1]
        # if file name is for exampel name.name2.extension
        elif len(full_file_name.split('.')) == 3:
            file_name = ''.join((full_file_name.split('.')[0],full_file_name.split('.')[1]))
            file_extension = full_file_name.split('.')[2]

        # if extension does not match, delete automatically
        if file_extension not in ['dat','rst','mac']:
            print("Removing file:",full_file_name)
            os.remove(''.join((analysis_path,'/',full_file_name)))
        # if extension matches
        else:
            try: # if last character of file name is a digit (e.g. file0.rst), delete file
                last_char = int(file_name[-1])
                print("Removing file:", full_file_name)
                os.remove(''.join((analysis_path, '/', full_file_name)))
                # else do nothing and move on to next file
            except:
                continue

class Parametric_study:
    def __init__(self, models:list, extraction_function = "MODAL"):
        """
        Launch and stores a parametric study with a list of models

        Attributes
        ----------
        models( list ): 
            List of models used for the parametric study

        results( list ):
            List of result class obtained after the parametric study
        
        model_dic( dict ):
            The keys are the models and the value the result
        
        extraction_function( func ):
            The function to extract the results
                results_class = extraction_function(PARAM_FOLDER, result_file, data_file,
                         saving_type="result-class", result_propreties=result_propreties,
                         save = False, file_name_full=full_file, mapdl=mapdl)
        
        """
        if extraction_function == "MODAL":
            self.extraction_function = ERA.extract_save_modal
        self.launch_simulations(models, False)
        model_dic = {}
        #print("number of final models :", len(self.models)) # returns 10
        #print("number of final results :", len(self.results))
        for model, result in zip(self.models, self.results):
            model_dic[model] = result
        self.model_dic = model_dic
        

    def launch_simulations(self, new_models, add_model = True):
        PARAM_FOLDER.mkdir(parents = True, exist_ok = True)
        mapdl = launch_mapdl(run_location=str(PARAM_FOLDER), mode='grpc', override=True)
        results = []
        working_models = []
        for model in tqdm(new_models):
            model:Fe_model
            if model.csv_file_path != None:
                model.write_mean_flow_csv_file()
            model.write_mapdl_file(str(PARAM_FOLDER), verbose = False)
            model_name = model.model_name
            db_name = model.db_name
            result_file = PARAM_FOLDER / (db_name + ".rst")
            data_file = PARAM_FOLDER / (db_name + ".dat")
            full_file = db_name + ".full"
            mapdl.input(model_name)
            result_propreties = model.result_propreties
            try:
                result = self.extraction_function(PARAM_FOLDER, result_file, data_file,
                         saving_type="result-class", result_propreties=result_propreties,
                         save = False, file_name_full=full_file, mapdl=mapdl)
                results.append(result)
                working_models.append(model)
            except:
                print("------------------------------------------")
                print("-----EXTRACT PROB WITH PARAMETERS :-------")
                pp.pprint(model.__str__())


        mapdl.exit()
        if add_model:
            self.models += working_models
            self.results += results
        else:
            self.models = working_models
            self.results = results
        return results

    def n_models(self):
        return len(self.results)

    def variable_list(self, variable_name:str, propreties_type:str = "fluid",):
        """
        To create a list for a given variable 
        """
        X = []
        for model in self.models:
            variables = model.variables
            fluid_propreties = model.fluid_propreties
            structural_propreties = model.structural_propreties
            variables_ = vars(model)
            if variable_name in variables.keys():
                X.append(variables[variable_name])
            elif variable_name in variables_.keys():
                X.append(variables_[variable_name])
            elif variable_name in fluid_propreties.keys() and propreties_type == "fluid":
                X.append(fluid_propreties[variable_name])
            elif variable_name in structural_propreties.keys() and propreties_type == "structural":
                X.append(structural_propreties[variable_name])
        return X
    

class Mean_Flow_Parametric_study(Parametric_study):
    def __init__(self, models:list, extraction_function = "MODAL"):

        self.models = models

        super().__init__(models, extraction_function)


    def launch_simulations(self, new_models, add_model = True):
        PARAM_FOLDER.mkdir(parents = True, exist_ok = True)
        mapdl = launch_mapdl(run_location=str(PARAM_FOLDER), mode='grpc', override=True)
        results = []
        working_models = []
        for model in tqdm(new_models):
            model:Disc_model
            model.write_mean_flow_csv_file()
            model.write_mapdl_file(str(PARAM_FOLDER), verbose = False)
            model_name = model.model_name
            db_name = model.db_name
            result_file = PARAM_FOLDER / (db_name + ".rst")
            data_file = PARAM_FOLDER / (db_name + ".dat")
            full_file = db_name + ".full"
            mapdl.input(model_name)
            result_propreties = model.result_propreties
            try:
                result = self.extraction_function(PARAM_FOLDER, result_file, data_file,
                            saving_type="result-class", result_propreties=result_propreties,
                            save = False, file_name_full=full_file, mapdl=mapdl)
                results.append(result)
                working_models.append(model)
            except:
                print("------------------------------------------")
                print("-----EXTRACT PROB WITH PARAMETERS :-------")
                pp.pprint(model.__str__())
                
        mapdl.exit()
        if add_model:
            self.models += working_models
            self.results += results
        else:
            self.models = working_models
            self.results = results
        return results


        
        
