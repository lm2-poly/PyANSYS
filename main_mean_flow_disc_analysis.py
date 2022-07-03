"""
Modification du conde main_disc_analysis.py pour porendre en compte la présence d'un meanf flow ou non.
Travail sur un modèle 3D, en plus du modèle 2D.

"""



#gestion des fichiers :
import pathlib
import os

#création de commandes terminal :
import argparse

#bibliothèque numpy et matplotlib :
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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

WATERFLOW = 2
WATER = 1
NOWATER = 0

#------------------------- Argument parser -------------------------------
#-------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--main_case')
parser.add_argument('--point_number', type = int, default=20)
parser.add_argument('--is_unconfined', action='store_true', default=False)
parser.add_argument('--mean_flow', type = str, help = "indicates which kind of mean flow is applied ; useable only with water around the disc. Choice between 'constant_omega' - 'variable_omega' - 'couette_flow' - 'Blais'")

args = parser.parse_args()

print("arguments of the parser :")
pp.pprint(vars(args))


if args.is_unconfined:
    analysis_type = "disc_unconfined"
else:
    analysis_type = "disc_confined"

mean_flow = False
CSV_file = False

if args.mean_flow == "constant_omega":
    flow_type = "cst_w"
    mean_flow = True
    CSV_file = True
elif args.mean_flow == "variable_omega":
    flow_type = "variable_omega"
    mean_flow = True
    CSV_file = True
elif args.mean_flow == "couette_flow":
    flow_type = "TC_flow"
    mean_flow = True
    CSV_file = True
elif args.mean_flow == "Blais":
    flow_type = "Blais_flow"
    mean_flow = True 

n_points = args.point_number

#------------------------- The file paths --------------------------------
#-------------------------------------------------------------------------
print("------------init file paths--------------")
print("-----------------------------------------")
working_folder = pathlib.Path(os.getcwd())
param_folder = working_folder / "parametric-study"
param_folder.mkdir(parents = True, exist_ok = True)
print("analyse type :")
print(analysis_type)

saving_folder = working_folder / "saving_folder/disc_param_study" / analysis_type

model_name = {NOWATER:"disc_no_water.mac", WATER:"disc.mac", WATERFLOW:"disc_mean_flow.mac"}

db_name = {NOWATER:"disc_no_water", WATER:"disc", WATERFLOW:"disc_mean_flow"}

template_file_path = {NOWATER:working_folder / ("template-apdl-disc" +  
                "/" + analysis_type + "/" + "/disc_no_water.mac"),
                    WATER:working_folder / ("template-apdl-disc" +  
                "/" + analysis_type + "/" + "/disc.mac"),
                    WATERFLOW:working_folder / ("template-apdl-disc" +  
                "/" + analysis_type + "/" + "/disc_mean_flow.mac")}



#useful only for mean flow simulation
if  CSV_file == True :
    csv_file_path = working_folder / ("csv_file" + "/" + flow_type)
else:
    csv_file_path = None





result_file = {NOWATER:db_name[NOWATER] + ".rst",
               WATER: db_name[WATER] + ".rst",
               WATERFLOW: db_name[WATERFLOW] + ".rst"}
result_file = {NOWATER: param_folder / result_file[NOWATER],
               WATER: param_folder / result_file[WATER],
               WATERFLOW: param_folder / result_file[WATERFLOW]}         

data_file = {NOWATER:db_name[NOWATER] + ".dat",
               WATER: db_name[WATER] + ".dat",
               WATERFLOW: db_name[WATERFLOW] + ".dat"}
data_file = {NOWATER: param_folder / data_file[NOWATER],
               WATER: param_folder / data_file[WATER],
               WATERFLOW: param_folder / data_file[WATERFLOW]}        

#------------------------- Variables name --------------------------------
#-------------------------------------------------------------------------
print("----------init variables name------------")
print("-----------------------------------------")

inner_radius = "inner_radius"
outer_radius="outer_radius"
spring_length="spring_length"
width_structure = "width_structure"
outer_water_radius = "outer_water_radius"
acoustic_radial_division = "acoustic_radial_division"
structural_radial_division = "structural_radial_division"
angular_division = "angular_division"
disc_width = "disc_width"
omega = "omega"
DB_NAME = "db_name"
tol="tol"
r1 = "real_constant_1" #spring stiffness
r2 = "real_constant_2" #reference pressure for acoustic_element
r3 = "real_constant_3" #mass value
r4 = "real_constant_4" #RAD, X0, Y0 for FLUID129
r5 = "real_constant_5" #plane thickness

if __name__ == "__main__":
    if args.main_case == "parametric_study":
        print("----" + "case : " + args.main_case + "----")
        print("----beginning of parametric study -------")
        print("-----------------------------------------")
#------------------------- Variables init --------------------------------
#-------------------------------------------------------------------------
        print("---- initialisation of the variables-----")
        print("-----------------------------------------")
        #variables with water but no flow:
        variables_no_flow = {}
        variables_no_flow[inner_radius] =2.5
        variables_no_flow[outer_radius] = 3.0 #2*variables_no_flow[inner_radius] # used only in the case of a confined disc
        variables_no_flow[width_structure] = 0.1
        variables_no_flow[spring_length] = 3.1
        variables_no_flow[acoustic_radial_division] = 5
        variables_no_flow[structural_radial_division] = 1
        variables_no_flow[angular_division] = 20
        variables_no_flow[tol]=0.002
        variables_no_flow[DB_NAME] = "'" + db_name[WATER] + "'"

        #variables with water and a flow:
        variables_flow = variables_no_flow.copy()
        variables_flow[disc_width] = 0.05
        variables_flow[DB_NAME] = "'" + db_name[WATERFLOW] + "'"

        #variables without water :
        variables_no_water = variables_no_flow.copy()
        variables_no_water[DB_NAME] = "'" + db_name[NOWATER] + "'"
        variables = {NOWATER:variables_no_water, WATER:variables_no_flow, WATERFLOW: variables_flow}

        #other variables :
        mass_element = "MASS21"
        spring_element = "COMBIN14"
        fluid_element = "FLUID29"
        fluid_element_3D = "FLUID220"
        fluid_infinite_element = "FLUID129"
        solid_element = "PLANE182"
        solid_element_3D = "SOLID186"
        sound_celerity = 1500
        structural_sound_celerity = 5000
        structural_density = 7200
        young_modulus = 289000000000
        poisson = 0.21
        fluid_densitys = np.linspace(0,2000,n_points)
        omegas = np.arange(0,30,3)
        real_constants={}
        real_constants[r1]=200000 #spring_stiffness
        real_constants[r2]=200 #reference pessure for acoustic_element
        real_constants[r3]=2e-5 #mass
        real_constants[r4]=4
        real_constants[r5]=0.001

#------------------------- Result propreties -----------------------------
#-------------------------------------------------------------------------
        print("---- initialisation of the results propreties----")
        print("-------------------------------------------------")
        #result propreties with water and no flow:
        n_modes = 4
        modes_to_extract = [i for i in range(n_modes)]
        if args.is_unconfined:
            n_ddl = 9
        else:
            n_ddl = 7
        disp_ddls = [0,1,2]
        press_ddls = [3]
        fluid_comp = FLUID_COMP
        solid_comp = SOLID_COMP
        result_propreties_no_flow = ARA.Result_propreties(modes_to_extract, n_ddl, solid_comp, fluid_comp, 
                                                disp_ddls, press_ddls)

        #result propreties without water :
        result_propreties_0 = deepcopy(result_propreties_no_flow)
        n_modes = 1
        modes_to_extract = [i for i in range(n_modes)]
        result_propreties_0.n_ddl = 7
        result_propreties_0.press_ddls = None
        result_propreties_0.modes_to_extract = modes_to_extract


        #result propreties with water and a flow:
        n_modes_case1 = 6
        modes_to_extract_case1 = [i for i in range(n_modes_case1)]

        n_modes_case2 = 16
        modes_to_extract_case2 = [i for i in range (n_modes_case2)]

        if args.is_unconfined:
            n_ddl = 9
        else:
            n_ddl = 7
        disp_ddls = [0,1,2]

        press_ddls = [3]
        fluid_comp = FLUID_COMP
        solid_comp = SOLID_COMP
        result_propreties_flow_case1 = ARA.Result_propreties(modes_to_extract_case1, n_ddl, solid_comp, fluid_comp, 
                                                disp_ddls, press_ddls)
        result_propreties_flow_case2 = ARA.Result_propreties(modes_to_extract_case2, n_ddl, solid_comp, fluid_comp, 
                                                disp_ddls, press_ddls)
        #case with 6 modes to extract
        result_propreties_case1 = {NOWATER:result_propreties_0, WATER:result_propreties_no_flow, WATERFLOW:result_propreties_flow_case1}
        #case with 16 modes to extract
        result_propreties_case2 = {NOWATER:result_propreties_0, WATER:result_propreties_no_flow, WATERFLOW:result_propreties_flow_case2}

#---------------------- Setting of models list ---------------------------
#-------------------------------------------------------------------------            
        print("---------- Setting of models list ---------")
        print("-------------------------------------------")
        # for cases with no interest in the mean flow effect, parametric study against the fluid density
        if mean_flow == False: 
            models = []
            water = NOWATER
            for i, fluid_density  in enumerate(fluid_densitys):
                if i > 0:
                    water = WATER
                model = LAS.Disc_model(model_name[water], db_name[water], 
            template_file_path[water], variables[water], fluid_element,
            solid_element, fluid_density, sound_celerity, structural_density,
            young_modulus, poisson, result_propreties_case1[water], real_constants, 
            mass_element, spring_element, fluid_infinite_element, 
            structural_sound_celerity)
                models.append(model)
        # for cases with mean flow effect, parametric study against the fluid angluar velocity
        else:
            fluid_density = 1000.0
            models = []
            water = WATERFLOW
            #we create 1 model per omega value
            for i, omega in enumerate(omegas):
                #print("MY OMEGA :",omega)   
                if (omega >10):
                    model = LAS.Disc_model(model_name[water], db_name[water], 
                template_file_path[water], variables[water], fluid_element,
                solid_element, fluid_density, sound_celerity, structural_density,
                young_modulus, poisson, result_propreties_case2[water], real_constants, 
                mass_element, spring_element, fluid_infinite_element, 
                structural_sound_celerity, dimension_3D = True, mean_flow = True, CSV_file = CSV_file,
                fluid_element_3D = fluid_element_3D, structural_element_3D = solid_element_3D,
                csv_file_path = csv_file_path, flow_type = flow_type, omega = omega) 

                else:
                    model = LAS.Disc_model(model_name[water], db_name[water], 
                template_file_path[water], variables[water], fluid_element,
                solid_element, fluid_density, sound_celerity, structural_density,
                young_modulus, poisson, result_propreties_case1[water], real_constants, 
                mass_element, spring_element, fluid_infinite_element, 
                structural_sound_celerity, dimension_3D = True, mean_flow = True, CSV_file = CSV_file,
                fluid_element_3D = fluid_element_3D, structural_element_3D = solid_element_3D,
                csv_file_path = csv_file_path, flow_type = flow_type, omega = omega)            

                models.append(model)

            

#---------------------- The parametric study ---------------------------
#----------------------------------------------------------------------- 
        print("------------- Parametric study ------------")
        print("-------------------------------------------")
        if CSV_file:
            parametric_study = LAS.Mean_Flow_Parametric_study(models)
        else:
            parametric_study = LAS.Parametric_study(models)
        
#---------------------- Saving of the model ---------------------------
#----------------------------------------------------------------------
        print("------------ Saving the study :------------")
        print("-------------------------------------------")
        ERA.save_object(saving_folder, parametric_study)
        print("saved in :", saving_folder)


    if args.main_case == "analyse-mean_flow_effect":
        print("----" + "case : " + args.main_case + " ----")
        parametric_result:LAS.Parametric_study = ERA.load_object(saving_folder)
        n_models = parametric_result.n_models()
        print("nombre de modèles :", n_models)
        models = parametric_result.models
        results = parametric_result.results
        omegas = parametric_result.variable_list("fluid_omega")
        print("the omegas :",omegas)
        natural_frequencies = []
        final_omegas = []
        frequency_j = 0
        #extraction of the natural frequencies for various omega
        for i, (model, result) in enumerate(zip(models, results)):
            #declare the model and the result to help 
            #to write the code (autocompletion)
            model:LAS.Disc_model
            result:ARA.Modal_result
            frequencies, ids = result.get_frequencies(decimal = 4, ids_ = True)
            omega = omegas[i]
            
            # case of a constant angular velocity omega
            if mean_flow:
                if flow_type == "cst_w":
                    title=("u_theta $= r\omega$, at $R_1 = 2.5$ m, $R_2 = 3$ m")

                #case of a Taylor-Couette flow:
                if flow_type == "TC_flow":
                    title="Taylor-Couette flow, at $R_2 = 3.5$ m"

                #test with flow of Blais
                if flow_type == "Blais_flow":
                    title="u_theta $= r\omega$, at $R_2 = 3$ m"


                #calculation of the natural frequencies
                print("OMEGA :",omega)
                for frequency in frequencies :
                    print(frequency)
                frequency_i = round(frequencies[0],5)
                k =0
                while(k<len(frequencies) and frequency_i<10):
                    k+=1
                    if k<len(frequencies):
                        frequency_i=round(frequencies[k],5)
                if i == 0:
                    frequency_j = frequency_i
                    print("FREQ 0:", frequency_j)
                    natural_frequencies.append(frequency_j)
                    final_omegas.append(omega)
                    print("iteration i :", omega, frequency_j)
                else:
                    # frequency_i = result.get_frequency(frequency_j,id)[0]
                    if (frequency_j-frequency_i)<10:
                    #     if (frequency_j > frequency_i):                
                        frequency_j = frequency_i
                        print("FREQ J",frequency_j)
                        natural_frequencies.append(frequency_j)
                        final_omegas.append(omega)
                        print("iteration i :", omega, frequency_j)

                
            print("frequencies : ",natural_frequencies)
            print("omegas :", final_omegas)


        #fluid_densitys = parametric_result.variable_list("density")
        fig = plt.figure()
        ax = fig.add_subplot(111) 
        plt.plot(final_omegas,natural_frequencies,linestyle = "none", marker = ".", label = "arange(0,30,3)")              
        plt.legend()
        plt.title(title)
        plt.xlabel("$\Omega$ (rad/s)")
        plt.ylabel("$f_n$ (Hz)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x}'))
        plt.show()

        models = parametric_result.models
