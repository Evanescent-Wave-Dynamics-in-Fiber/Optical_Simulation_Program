#Minimum Angle of Incidence = 75 degree

#The results are divided into two distinct regions based on the parameter p: 
#the first region spans from 0.0001 to 0.15, 
#while the second region extends from 0.15 to 1.00. 
#And in both regions, the characteristic length scale Za ranges from 1 nm to 200 nm 

# ----------------------------------------------------------------------------
# import module
# ----------------------------------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import TriAnalyzer, Triangulation, UniformTriRefiner
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd
import math

from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_rgb, to_hex

# Interpolation function: Generate n transitional colors between two colors
def interpolate_colors(color1, color2, n):
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    return [to_hex(rgb1 + (rgb2 - rgb1) * j / (n - 1)) for j in range(n)]


# ----------------------------------------------------------------------------
# import custom module (in Subfunction branch)
# ----------------------------------------------------------------------------

from Function_Code import Function_for_Creating_Dataform_Delaunay_Need as f
from Function_Code import Function_for_Model as f1
from Function_Code import Function_for_Creating_RSM_lattice as get_point 
from Function_Code import Function_for_Add_boundary as Add_boundary

# ----------------------------------------------------------------------------
# Set parameters
# ----------------------------------------------------------------------------

increament=0.0035 # The interval of angle in Model

import os

project_root = os.path.dirname(os.path.abspath(__file__))
excel_folder = os.path.join(project_root, 'Excel')
font_folder = os.path.join(project_root, 'Font_Calibri')

experimental_file=os.path.join(excel_folder, 'Excel for Refractive Index of Catalysts.xlsx')

Calibri_pathname=os.path.join(font_folder, 'calibri.ttf')
Calibrii_pathname=os.path.join(font_folder, 'calibrii.ttf')

font_properties=FontProperties(fname=Calibri_pathname, size=10)
font_properties_ii=FontProperties(fname=Calibrii_pathname, size=10)

#First data transition file address
workbook_file1=os.path.join(excel_folder, 'Excel for Workbook File 1_pza_RSM_boundary1.xlsx')
workbook_file_delaunay1=os.path.join(excel_folder, 'Excel for Workbook File 1_pza_RSM_boundary1_delaunay1.xlsx')

#Second data transition file address
workbook_file2=os.path.join(excel_folder, 'Excel for Workbook File 2_pza_RSM_boundary2.xlsx')
workbook_file_delaunay2=os.path.join(excel_folder, 'Excel for Workbook File 2_pza_RSM_boundary1_delaunay2.xlsx')

# ----------------------------------------------------------------------------
# Set getting points' parameter(First)
# ----------------------------------------------------------------------------

#The first Region
p_x1 =0.0001
#za_y1 =0.000000001
za_y1 =0.000000000001

interval1_x1=0.16
interval2_y1=0.000000200

#Set Mesh Space
n_x1= 15
m_y1= 40

# First Delaunay parameters
subdiv_1 =0
init_mask_frac_1 = 0
min_circle_ratio_1 = 0.04

# ----------------------------------------------------------------------------
# Set getting points' parameter(Second)
# ----------------------------------------------------------------------------

#The Second Region
p_x2 =0.15
#za_y2 =0.000000001
za_y2 =0.00000000000001

interval1_x2=0.85
interval2_y2=0.000000200

#Set Mesh Space
n_x2= 15
m_y2= 15

#Second Delaunay parameters
subdiv_2 =0
init_mask_frac_2 = 0
min_circle_ratio_2 = 0.04

# ----------------------------------------------------------------------------
# Create two new Excel files to store the simulation data.
# ----------------------------------------------------------------------------

def create_excel_xls(path):
    data_df = pd.DataFrame()
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer)
    writer.close()

create_excel_xls(workbook_file1)
writer1=pd.ExcelWriter(workbook_file1)

create_excel_xls(workbook_file2)
writer2=pd.ExcelWriter(workbook_file2)

# ----------------------------------------------------------------------------
# Generate points (First)
# ----------------------------------------------------------------------------

interval_pacthiness = interval1_x1/n_x1
interval_za = interval2_y1/m_y1

point1 = []
point2 = []

for i in range(n_x1) :
    
    for j in range(m_y1):
    
        point11 = get_point.get_RSM_point1(p_x1 + i * interval_pacthiness , p_x1 + (i+1) * interval_pacthiness  )
        point1 = point1 + point11
        
        point22 = get_point.get_RSM_point2(za_y1 + j * interval_za , za_y1 + (j+1) * interval_za  )
        point2 = point2 + point22

boundary=Add_boundary.boundary_point(p_x1,p_x1+interval1_x1,za_y1,za_y1+interval2_y1,interval_pacthiness,interval_za)
point1=point1+boundary[0]+[p_x1+interval1_x1]
point2=point2+boundary[1]+[za_y1+interval2_y1]


df=pd.DataFrame({'patchiness':point1 , 'za':point2})

df.to_excel(writer1)
writer1.close()

# ----------------------------------------------------------------------------
# Generate points (Second)
# ----------------------------------------------------------------------------

interval_pacthiness = interval1_x2/n_x2
interval_za = interval2_y2/m_y2

point1 = []
point2 = []

for i in range(n_x2) :
    
    for j in range(m_y2):
    
        point11 = get_point.get_RSM_point1(p_x2 + i * interval_pacthiness , p_x2 + (i+1) * interval_pacthiness  )
        point1 = point1 + point11
        
        point22 = get_point.get_RSM_point2(za_y2 + j * interval_za , za_y2 + (j+1) * interval_za  )
        point2 = point2 + point22

boundary=Add_boundary.boundary_point(p_x2,p_x2+interval1_x2,za_y2,za_y2+interval2_y2,interval_pacthiness,interval_za)
point1=point1+boundary[0]+[p_x2+interval1_x2]
point2=point2+boundary[1]+[za_y2+interval2_y2]


df=pd.DataFrame({'patchiness':point1 , 'za':point2})

df.to_excel(writer2)
writer2.close()

# ----------------------------------------------------------------------------
# Get Result Excel for Delaunay (prepare parameters)
# ----------------------------------------------------------------------------

#Define exper variable
medium_refraction_index=[]
length=[]
wavelength=[]
diameter=[]
incident_energy=[]
fiber_index=[]
nT=[]
lamp_radius=[]
max_distance=[]
k=[]

df=pd.read_excel(experimental_file)
number=len(df.values) #The number of Exper
for i in range(number):
    medium_refraction_index.append(df.values[i,0])
    length.append(df.values[i,1])
    diameter.append(df.values[i,2])
    wavelength.append(df.values[i,3])
    incident_energy.append(df.values[i,4])
    fiber_index.append(df.values[i,5])
    nT.append(df.values[i,6])
    lamp_radius.append(df.values[i,7])
    max_distance.append(df.values[i,8])
    k.append(df.values[i,9])

df1=pd.read_excel(workbook_file1) 
df2=pd.read_excel(workbook_file2) 

# ----------------------------------------------------------------------------
# Get Result Excel for Delaunay (First)
# ----------------------------------------------------------------------------

p=[]
za=[]
create_excel_xls(workbook_file_delaunay1)
writer=pd.ExcelWriter(workbook_file_delaunay1)
m=len(df1.values)
for i in range(m): 
    p.append(df1.values[i,1])
    za.append(df1.values[i,2])
for i in [0]:  
    minangle=1.30899693
    maxangle=1.569
    fiber_critical_angle=f.get_fiber_critical_angle(fiber_index[i], medium_refraction_index[i])
    n=f.calculate_angle_divided_number(minangle, maxangle, increament)
    incident_energy[i]=f1.chageincident_energy_simple(incident_energy[i], n)
    angle=f.get_corresponding_angle(n, minangle, maxangle, fiber_critical_angle, increament)
    edeep=f.calculate_edeep(fiber_index[i], angle, medium_refraction_index[i], wavelength[i])
    Tqt=f.calculate_Tqt(n, fiber_index[i], angle, nT[i])
    Tmedium=f.calculate_Tmedium(medium_refraction_index[i], fiber_index[i], angle)
    N=f1.calculate_reflection_number_simple(n,angle, length[i], diameter[i])
    z=f.calculate_intermediate_quantity_z(m, angle, p, za, length[i], diameter[i], Tqt, incident_energy[i], edeep, N)
    energy_edisspated=f.calculate_energy_edisspated_list(m, angle, incident_energy[i], z)
    energy_rdisspated=f.calculate_energy_rdisspated_list(m, angle, incident_energy[i], z)
    energy_nontir=f.calculate_energy_nontir(m, angle, p, Tqt, incident_energy[i], N, Tmedium)
    disspated_energy=f1.sum_disspated_energy_123_simple(energy_edisspated, energy_rdisspated, m,energy_nontir)
    disspated_radiao=f1.calculate_radio_13_mode123_simple(m,n, incident_energy[i] , disspated_energy)
    radio_e=f1.calculate_radio_e_simple(energy_edisspated,m, incident_energy[i],disspated_energy)
    radio_r=f1.calculate_radio_r_simple(energy_rdisspated, m, incident_energy[i],disspated_energy)
    absorb_coefficient=f.calculate_absorb_coefficient(wavelength[i],k[i],0.0000001)
    near_field_ratio=[radio_e[k]+radio_r[k]*absorb_coefficient for k in range(m)]
    print('1down')
    
    df=pd.DataFrame({'patchiness':p,'za':za,'E_edis_ratio':radio_e[i]})
    df.to_excel(writer)
    writer.close()



# ----------------------------------------------------------------------------
# Get Result Excel for Delaunay (Second)
# ----------------------------------------------------------------------------

p=[]
za=[]
create_excel_xls(workbook_file_delaunay2)
writer=pd.ExcelWriter(workbook_file_delaunay2)
m=len(df2.values)
for i in range(m): 
    p.append(df2.values[i,1])
    za.append(df2.values[i,2])
for i in [0]:  
    minangle=1.30899693
    maxangle=1.569
    fiber_critical_angle=f.get_fiber_critical_angle(fiber_index[i], medium_refraction_index[i])
    n=f.calculate_angle_divided_number(minangle, maxangle, increament)
    incident_energy[i]=f1.chageincident_energy_simple(incident_energy[i], n)
    angle=f.get_corresponding_angle(n, minangle, maxangle, fiber_critical_angle, increament)
    edeep=f.calculate_edeep(fiber_index[i], angle, medium_refraction_index[i], wavelength[i])
    Tqt=f.calculate_Tqt(n, fiber_index[i], angle, nT[i])
    Tmedium=f.calculate_Tmedium(medium_refraction_index[i], fiber_index[i], angle)
    N=f1.calculate_reflection_number_simple(n,angle, length[i], diameter[i])
    z=f.calculate_intermediate_quantity_z(m, angle, p, za, length[i], diameter[i], Tqt, incident_energy[i], edeep, N)
    energy_edisspated=f.calculate_energy_edisspated_list(m, angle, incident_energy[i], z)
    energy_rdisspated=f.calculate_energy_rdisspated_list(m, angle, incident_energy[i], z)
    energy_nontir=f.calculate_energy_nontir(m, angle, p, Tqt, incident_energy[i], N, Tmedium)
    disspated_energy=f1.sum_disspated_energy_123_simple(energy_edisspated, energy_rdisspated, m,energy_nontir)
    disspated_radiao=f1.calculate_radio_13_mode123_simple(m,n, incident_energy[i] , disspated_energy)
    radio_e=f1.calculate_radio_e_simple(energy_edisspated,m, incident_energy[i],disspated_energy)
    radio_r=f1.calculate_radio_r_simple(energy_rdisspated, m, incident_energy[i],disspated_energy)
    absorb_coefficient=f.calculate_absorb_coefficient(wavelength[i],k[i],0.0000001)
    near_field_ratio=[radio_e[k]+radio_r[k]*absorb_coefficient for k in range(m)]
    print('2down')
    
    df=pd.DataFrame({'patchiness':p,'za':za,'E_edis_ratio':radio_e[i]})
    df.to_excel(writer)
    writer.close()
    
# ----------------------------------------------------------------------------
# Get Result Excel for Delaunay (First)
# ----------------------------------------------------------------------------

df=pd.read_excel(workbook_file_delaunay1)

# Initial points
x_initial=df['patchiness'].values
y_initial=df['za'].values
z_initial=df['E_edis_ratio'].values

x=np.array(x_initial)
y=np.array(y_initial)
z=np.array(z_initial)

# meshing with Delaunay triangulation
tri = Triangulation(x, y)
ntri = tri.triangles.shape[0]

# masking badly shaped triangles at the border of the triangular mesh.
mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio_1)
tri.set_mask(mask)

# refining the data
refiner = UniformTriRefiner(tri)
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=subdiv_1)

# Output a new xy lattice
p1=tri_refi.x
za1=tri_refi.y

m1=len(p1)  
print(m1)

# ----------------------------------------------------------------------------
# Get Result Excel for Delaunay (Second)
# ----------------------------------------------------------------------------

df=pd.read_excel(workbook_file_delaunay2)

# Initial points
x_initial=df['patchiness'].values
y_initial=df['za'].values
z_initial=df['E_edis_ratio'].values

x=np.array(x_initial)
y=np.array(y_initial)
z=np.array(z_initial)

# meshing with Delaunay triangulation
tri = Triangulation(x, y)
ntri = tri.triangles.shape[0]

# masking badly shaped triangles at the border of the triangular mesh.
mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio_2)
tri.set_mask(mask)

# refining the data
refiner = UniformTriRefiner(tri)
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=subdiv_2)

# Output a new xy lattice
p2=tri_refi.x
za2=tri_refi.y

m2=len(p2)  
print(m2)

# ----------------------------------------------------------------------------
# Get data
# ----------------------------------------------------------------------------

#Correct the Units
za11=za1*1000000000
za22=za2*1000000000
tri1=Triangulation(p1, za11)
tri2=Triangulation(p2, za22)
print(za11)
print(za1)



# ----------------------------------------------------------------------------
# Calculation(First)(Second)
# ----------------------------------------------------------------------------

for i in range(number):  
    minangle=1.30899693
    maxangle=1.569
    fiber_critical_angle=f.get_fiber_critical_angle(fiber_index[i], medium_refraction_index[i])
    n=f.calculate_angle_divided_number(minangle, maxangle, increament)
    incident_energy[i]=f1.chageincident_energy_simple(incident_energy[i], n)
    angle=f.get_corresponding_angle(n, minangle, maxangle, fiber_critical_angle, increament)
    edeep=f.calculate_edeep(fiber_index[i], angle, medium_refraction_index[i], wavelength[i])
    Tqt=f.calculate_Tqt(n, fiber_index[i], angle, nT[i])
    Tmedium=f.calculate_Tmedium(medium_refraction_index[i], fiber_index[i], angle)
    N=f1.calculate_reflection_number_simple(n,angle, length[i], diameter[i])
    absorb_coefficient=f.calculate_absorb_coefficient(wavelength[i],k[i],0.0000001)
    
    z=f.calculate_intermediate_quantity_z(m1, angle, p1, za1, length[i], diameter[i], Tqt, incident_energy[i], edeep, N)
    energy_edisspated=f.calculate_energy_edisspated_list(m1, angle, incident_energy[i], z)
    energy_rdisspated=f.calculate_energy_rdisspated_list(m1, angle, incident_energy[i], z)
    energy_nontir=f.calculate_energy_nontir(m1, angle, p1, Tqt, incident_energy[i], N, Tmedium)
    del z
    disspated_energy=f1.sum_disspated_energy_123_simple(energy_edisspated, energy_rdisspated, m1,energy_nontir)
    disspated_radiao1=f1.calculate_radio_13_mode123_simple(m1,n, incident_energy[i] , disspated_energy)
    radio_e1=f1.calculate_radio_e_simple(energy_edisspated,m1, incident_energy[i],disspated_energy)
    radio_r1=f1.calculate_radio_r_simple(energy_rdisspated, m1, incident_energy[i],disspated_energy)
    del energy_edisspated
    del energy_rdisspated
    del energy_nontir
    del disspated_energy
    
    near_field_ratio1=[radio_e1[k]+radio_r1[k]*absorb_coefficient for k in range(m1)]
    print('{number}2down'.format(number=i+1))
    
    z=f.calculate_intermediate_quantity_z(m2, angle, p2, za2, length[i], diameter[i], Tqt, incident_energy[i], edeep, N)
    energy_edisspated=f.calculate_energy_edisspated_list(m2, angle, incident_energy[i], z)
    energy_rdisspated=f.calculate_energy_rdisspated_list(m2, angle, incident_energy[i], z)
    energy_nontir=f.calculate_energy_nontir(m2, angle, p2, Tqt, incident_energy[i], N, Tmedium)
    del z
    del N
    disspated_energy=f1.sum_disspated_energy_123_simple(energy_edisspated, energy_rdisspated, m2,energy_nontir)
    disspated_radiao2=f1.calculate_radio_13_mode123_simple(m2,n, incident_energy[i] , disspated_energy)
    radio_e2=f1.calculate_radio_e_simple(energy_edisspated,m2, incident_energy[i],disspated_energy)
    radio_r2=f1.calculate_radio_r_simple(energy_rdisspated, m2, incident_energy[i],disspated_energy)
    del energy_edisspated
    del energy_rdisspated
    del energy_nontir
    del disspated_energy
    
    near_field_ratio2=[radio_e2[k]+radio_r2[k]*absorb_coefficient for k in range(m2)]
    print('{number}2down'.format(number=i+1))
    
    #color of 1
    levels_1 = np.linspace(0, 1, 22)  
    colors_1 = ['#2C4B75', '#4575B4', '#91BFDB', '#E0F3F8', '#FEE090', '#FC8D59','#D73027','#9A221C']

    n_interpolate = 3  
    gradient_colors_1 = []
    for ccco in range(len(colors_1) - 1):
      gradient_colors_1.extend(interpolate_colors(colors_1[ccco], colors_1[ccco + 1], n_interpolate + 1)[:-1])  # 避免重复
    gradient_colors_1.append(colors_1[-1])  

    #color of NF
    levels_NF = np.linspace(0, 1, 25)  
    colors_NF = ['#F9E29E', '#FEC88C', '#FC9366', '#F0605D', '#CC3E70', '#9E2E7E','#711F81','#440F75','#282A62']

    n_interpolate = 3  
    gradient_colors_NF = []
    for ccco in range(len(colors_NF) - 1):
        gradient_colors_NF.extend(interpolate_colors(colors_NF[ccco], colors_NF[ccco + 1], n_interpolate + 1)[:-1])  # 避免重复
    gradient_colors_NF.append(colors_NF[-1])  


    # ----------------------------------------------------------------------------
    # Graph Z1
    # ----------------------------------------------------------------------------
    
    z1=disspated_radiao1
    del disspated_radiao1
    
    fig,ax=plt.subplots()
    
    cs=ax.tricontourf(tri1, z1, levels=levels_1,colors=gradient_colors_1)
    #operate X
    plt.xlim(0,1)
    plt.xlabel(r"$\mathit{p_c}$ (cm$^2$ cm$^{-2}$)", fontsize=10, fontproperties=font_properties)
    plt.ylabel(r"z$_a$ $\mathrm{(nm)}$",fontsize=10, fontproperties=font_properties_ii)

    # Set x-axis major ticks
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=10, fontproperties=font_properties)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='x', which='minor', length=3) 

    # Set y-axis major ticks
    ax.set_ylim(bottom=0)  # 
    plt.yticks([0, 40, 80, 120, 160, 200],fontsize=10, fontproperties=font_properties)
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.tick_params(axis='y', which='minor', length=3)

    
    z1=disspated_radiao2
    del disspated_radiao2
    
    cs=ax.tricontourf(tri2, z1, levels=levels_1,colors=gradient_colors_1)
    plt.title(f"$n_c$ = {nT[i]:.2f}", fontsize=10, fontproperties=font_properties)

    #set colorbar
    cbar=fig.colorbar(cs)

    pos = cbar.ax.get_position()  
    # Position
    new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
    cbar.ax.set_position(new_pos)
    
    colorbarticks=np.arange(0, 1.1, 0.2)
    cbar.set_ticks(colorbarticks,labels=['0','20','40','60','80','100'],fontsize=10, fontproperties=font_properties)
    txt=r'Φ$_{\mathrm{rad}}$/Φ$_{\mathrm{in}}$ $(\% \, \, \mathrm{cm}^{-1})$'
    cbar.ax.set_title(txt, fontsize=10, fontproperties=font_properties_ii,loc='center')
    
    for line in cbar.ax.get_children():
        if isinstance(line, plt.Line2D):
           line.set_alpha(0.5)  # transparency of line in cbar

    # ----------------------------------------------------------------------------
    # Save z1
    # ----------------------------------------------------------------------------
    
    plt.savefig(f'nc-rad-in{i+1}.jpg',format='jpg',dpi=300)
    
    # ----------------------------------------------------------------------------
    # Graph ZNF
    # ----------------------------------------------------------------------------

    zNF=near_field_ratio1
    del near_field_ratio1
    
    fig,ax=plt.subplots()

    cs = ax.tricontourf(tri1, zNF, levels=levels_NF, colors=gradient_colors_NF)

    #operate X
    plt.xlim(0,1)
    plt.xlabel(r"$p_c$ (cm$^2$ cm$^{-2}$)", fontsize=10, fontproperties=font_properties)
    plt.ylabel(r"z$_a$ $\mathrm{(nm)}$",fontsize=10, fontproperties=font_properties_ii)
    
    # Set x-axis major ticks
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=10, fontproperties=font_properties)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='x', which='minor', length=3)  

    # Set y-axis major ticks
    ax.set_ylim(bottom=0)  
    plt.yticks([0, 40, 80, 120, 160, 200],fontsize=10, fontproperties=font_properties)
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.tick_params(axis='y', which='minor', length=3)
    
    zNF=near_field_ratio2
    del near_field_ratio2
    
    cs = ax.tricontourf(tri2, zNF, levels=levels_NF, colors=gradient_colors_NF)
    plt.title(f"$n_c$ = {nT[i]:.2f}", fontsize=10, fontproperties=font_properties)

    #set colorbar
    cbar=fig.colorbar(cs)

    pos = cbar.ax.get_position()  
    # Position
    new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
    cbar.ax.set_position(new_pos)
    
    colorbarticks=np.arange(0, 1.1, 0.2)
    cbar.set_ticks(colorbarticks,labels=['0','20','40','60','80','100'],fontsize=10, fontproperties=font_properties)
    txt=r'Φ$_{\mathrm{NF}}$/Φ$_{\mathrm{rad}}$ $(\% \, \, \mathrm{cm}^{-1})$'
    cbar.ax.set_title(txt, fontsize=10, fontproperties=font_properties_ii,loc='center')
    
    for line in cbar.ax.get_children():
        if isinstance(line, plt.Line2D):
           line.set_alpha(0.5)  # transparency of line in cbar
    
    # ----------------------------------------------------------------------------
    # Save zNF
    # ----------------------------------------------------------------------------

    plt.savefig(f'nc-NF-rad{i+1}.jpg',format='jpg',dpi=300)

