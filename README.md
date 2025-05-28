**Optical Simulation Program**
---------------------------------------------------------------------------------------------------------------------------
Code for the manuscript "Facile-engineered metasurfaces modulate evanescent wave dynamics in near-fields of optical fibers for tackling sustainability challenges."

Author: Yinghao Song, Zexuan Ma, Zhe Zhao, Meng Zhang, Xinyu Chang, Han Wang, Meng Ni, Paul Westerhoff, Chii Shang, Li Ling

**Notation**
---------------------------------------------------------------------------------------------------------------------------
**FM-OFs** = Functional material coated optical fibers

**ONFs** = Optical near-fields

**$p_c$** = Surface patchiness (i.e., the fraction of materials in direct contact with fiber surfaces) 

**$z_a$** = Interspace distance

**$n_c$** = Refractive index of Coatings of designated functional materials

**$n_e$** = Refractive index of external bulk media

**$\theta_m$** = Minimum incident angle

**$\lambda$** = Light wavelength

**$Φ_{NF}/Φ_{rad}$×100%** = The percentage of the energy utilization in ONFs to total radial output

**$Φ_{rad}/Φ_{in}$×100%** = The percentage of the radial output relative to input light

**Code**
---------------------------------------------------------------------------------------------------------------------------
(1) Each script models how different parameters affect **$Φ_{NF}/Φ_{rad}$×100%** and **$Φ_{rad}/Φ_{in}$×100%** as functions of **$p_c$** and **$z_a$** for various **FM-OFs** configurations.:

| Script ID | Parameter Analyzed | Config File |
|-----------|--------------------|-------------|
| `Script_1` | Refractive index of external bulk media | `Excel for Refractive Index of External Medium.xlsx` |
| `Script_2` | Refractive index of Coatings of designated functional materials | `Excel for Refractive Index of Catalyst.xlsx` |
| `Script_3` | Light wavelength | `Excel for Wavelength.xlsx` |
| `Script_4` | Minimum incident angle | `Excel for Minimum Angle of Incidence.xlsx` |

<sub>**Script_1** = Code_for_Investigating_the_Impact_of_Refractive_Index_Changes_in_External_Media.py</sub>  
<sub>**Script_2** = Code_for_Investigating_the_Impact_of_Refractive_Index_Changes_in_Catalysts.py</sub>  
<sub>**Script_3** = Code_for_Investigating_the_Impact_of_Changes_in_Wavelength.py</sub>  
<sub>**Script_4** = Code_for_Investigating_the_Impact_of_Changes_in_Minimum_Angle_of_Incidence.py</sub>  

Flexible Parameter Setup – Simulate different conditions with ease:

- First: Open the relevant Excel file(Configuration File) in the `Excel` folder.
- Second: Adjust parameter values.
- Third: Re-run the corresponding script.

(2) The `Result` folder contains `.mov` format videos generated from `.jpg` images produced by the simulation program. These videos visually summarize the output under various simulation parameters.


**Python Dependencies**
---------------------------------------------------------------------------------------------------------------------------
- numpy==2.0.2
- pandas==2.2.2
- matplotlib==3.10.0
- scipy==1.15.3

**🛠 Environment**
---------------------------------------------------------------------------------------------------------------------------
- Platform: Google Colab
- Operating System: Linux (Kernel 6.1.123+)
- Python: 3.11.12


