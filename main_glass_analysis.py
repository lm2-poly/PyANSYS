#gestion des fichiers :
import pathlib
import os

#création de commandes terminal :
import argparse

#bibliothèque numpy :
import numpy as np

#importation de la bibliothèque créée :
import extract_result_ansys as ERA
import analyse_result_ansys as ARA
import launch_ansys_simulation as LAS

#Pour copier des variables :
from copy import deepcopy

#pour faire de jolis print :
import pprint as pp


#------------------------- Global constants ------------------------------
#-------------------------------------------------------------------------
print("----------init global constants----------")
print("-----------------------------------------")
FREQ = "freq"
COMP = "comp"
NODAL = "nodal-result"
POS = "nodal-position"
ID_INDEX = "id-to-index"
FREQ_INDEX = "freq-index"
SOLID_COMP = "SOLID_NODES"
FLUID_COMP = "FLUID_NODES"
WATER = 1
NOWATER = 0

#------------------------- Argument parser -------------------------------
#-------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--main_case')
parser.add_argument('--point_number', type = int, default=20)

args = parser.parse_args()

print("arguments of the parser :")
pp.pprint(vars(args))

n_points = args.point_number

#------------------------- The file paths --------------------------------
#-------------------------------------------------------------------------
print("------------init file paths--------------")
print("-----------------------------------------")
working_folder = pathlib.Path(os.getcwd())
param_folder = working_folder / "parametric-study"
param_folder.mkdir(parents = True, exist_ok = True)
saving_folder = working_folder / "saving_folder/glass_param_study"

model_name = {NOWATER:"cylinder_no_water.mac", WATER:"cylinder.mac"}

db_name = {NOWATER:"cylinder_no_water", WATER:"cylinder"}

template_file_path = {NOWATER:working_folder / "template-apdl-glass/cylinder_no_water.mac",
                      WATER:working_folder / "template-apdl-glass/cylinder.mac"}

result_file = {NOWATER:db_name[NOWATER] + ".rst",
               WATER: db_name[WATER] + ".rst"}
result_file = {NOWATER: param_folder / result_file[NOWATER],
               WATER: param_folder / result_file[WATER]}         

data_file = {NOWATER:db_name[NOWATER] + ".dat",
               WATER: db_name[WATER] + ".dat"}
data_file = {NOWATER: param_folder / data_file[NOWATER],
               WATER: param_folder / data_file[WATER]}        

#------------------------- Variables name --------------------------------
#-------------------------------------------------------------------------
print("----------init variables name------------")
print("-----------------------------------------")
average_radius = "average_radius"
width_structure = "width_structure"
outer_water_radius = "outer_water_radius"
radial_division = "radial_division"
angular_division = "angular_division"
DB_NAME = "db_name"
tol="tol"

if __name__ == "__main__":
    if args.main_case == "parametric_study":
        print("----beginning of parametric study -------")
        print("-----------------------------------------")
#------------------------- Variables init --------------------------------
#-------------------------------------------------------------------------
        print("---- initialisation of the variables-----")
        print("-----------------------------------------")
        #variables with water :
        variables = {}
        variables[average_radius] = 1.0
        variables[width_structure] = 0.1
        variables[outer_water_radius] = 2.0
        variables[radial_division] = 10 
        variables[angular_division] = 10
        variables[DB_NAME] = "'" + db_name[WATER] + "'"

        #variables without water :
        variables_no_water = variables.copy()
        variables_no_water[DB_NAME] = "'" + db_name[NOWATER] + "'"
        variables = {NOWATER:variables_no_water, WATER:variables}

        #other variables :
        fluid_element = "FLUID29"
        solid_element = "PLANE182"
        sound_celerity = 1430
        structural_density = 2500
        young_modulus = 70e9
        poisson = 0.22
        fluid_densitys = np.linspace(0,2000,n_points)

#------------------------- Result propreties -----------------------------
#-------------------------------------------------------------------------
        print("---- initialisation of the results propreties----")
        print("-------------------------------------------------")
        #result propreties with water :
        n_modes = 12
        modes_to_extract = [i for i in range(n_modes)]
        n_ddl = 3 #X, Y, pressure
        disp_ddls = [0,1]
        press_ddls = [2]
        fluid_comp = FLUID_COMP
        solid_comp = SOLID_COMP
        result_propreties = ARA.Result_propreties(modes_to_extract, n_ddl, solid_comp, fluid_comp, 
                                                disp_ddls, press_ddls)

        #result propreties without water :
        result_propreties_0 = deepcopy(result_propreties)
        n_modes = 7
        modes_to_extract = [i for i in range(n_modes)]
        result_propreties_0.n_ddl = 2
        result_propreties_0.press_ddls = None
        result_propreties_0.modes_to_extract = modes_to_extract
        result_propreties = {NOWATER:result_propreties_0, WATER:result_propreties}

#---------------------- Setting of models list ---------------------------
#-------------------------------------------------------------------------            
        print("---------- Setting of models list ---------")
        print("-------------------------------------------")
        models = []
        water = NOWATER
        for i, fluid_density in enumerate(fluid_densitys):
            if i > 0:
                water = WATER
            model = LAS.Glass_model(model_name[water], db_name[water], 
                        template_file_path[water], variables[water], fluid_element,
                        solid_element, fluid_density, sound_celerity,
                        structural_density, young_modulus, poisson, 
                        result_propreties[water])
            models.append(model)

#---------------------- The parametric study ---------------------------
#----------------------------------------------------------------------- 
        print("------------- Parametric study ------------")
        print("-------------------------------------------")
        parametric_study = LAS.Parametric_study(models)

#---------------------- Saving of the model ---------------------------
#----------------------------------------------------------------------
        print("------------ Saving the study :------------")
        print("-------------------------------------------")
        ERA.save_object(saving_folder, parametric_study)


        