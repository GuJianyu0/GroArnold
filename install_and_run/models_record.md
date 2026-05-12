## Here are the models we run and their parameter settings. One should reset parameters in the 4 files in this folder (IC_DICE_manucraft.params, run.param, IC_setting_list.txt, user_settings_multi.txt) when running each of the models. Note that it spend much time when running simulation, foci, actions and fitting. The 4 fils would be copy to the "history_runnings_XXX" folder after running.
## Check these "#cmd" values in run.bash file and then run in shell of this folder: $bash run.bash
## NFW: NFW profile; Ein: EinastoUsual profile. L: low value; H: high value. 

1. NFW_spinL_axisLH
"model1" in IC_DICE_manucraft.params (profile): 8
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile2
line1 in IC_setting_list.txt (model name): NFW_spinL_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
line4 in IC_setting_list.txt (powerlaw or Sersic): 0
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 1
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): NFW_spinL_axisLH0 NFW_spinL_axisLH1
"is_modify_IC", the 7th from 0 word, in user_settings_multi.txt (spin low: 0; spin high: 1): 0

2. NFW_spinH_axisLH
"model1" in IC_DICE_manucraft.params (profile): 8
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile2
line1 in IC_setting_list.txt (model name): NFW_spinH_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
line4 in IC_setting_list.txt (0 if fitting powerlaw or 1 if fitting Sersic): 0
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 1
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): NFW_spinH_axisLH0 NFW_spinH_axisLH1
"is_modify_IC" in user_settings_multi.txt (spin): 1

3. Ein_spinL_axisLH
"model1" in IC_DICE_manucraft.params (profile): 21
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile1
line1 in IC_setting_list.txt (model name): Ein_spinL_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
line4 in IC_setting_list.txt (powerlaw or Sersic): 1
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 1
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): Ein_spinL_axisLH0 Ein_spinL_axisLH1
"is_modify_IC" in user_settings_multi.txt (spin): 0

4. Ein_spinH_axisLH
"model1" in IC_DICE_manucraft.params (profile): 21
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile1
line1 in IC_setting_list.txt (model name): Ein_spinH_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
line4 in IC_setting_list.txt (powerlaw or Sersic): 1
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 1
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): Ein_spinH_axisLH0 Ein_spinH_axisLH1
"is_modify_IC" in user_settings_multi.txt (spin): 1

4-1. Ein_potI_spinH_axisLH
"model1" in IC_DICE_manucraft.params (profile): 21
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile1
line1 in IC_setting_list.txt (model name): Ein_spinH_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
the first value of line4 in IC_setting_list.txt (powerlaw or Sersic): 1
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 0
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): Ein_spinH_axisLH0 Ein_spinH_axisLH1
"is_modify_IC" in user_settings_multi.txt (spin): 1

4-2. Ein_multicomp_spinL_axisLH
"model1" in IC_DICE_manucraft.params (profile): 21
#?? python write settings file
uncomment these relative lines of code  in IC_DICE_manucraft.params (profile): profile1
line1 in IC_setting_list.txt (model name): Ein_multicomp_spinL_axisLH
line2 in IC_setting_list.txt (various parameter): flatz1
line3 in IC_setting_list.txt (values of various parameter): 0.3 0.6
the first value of line4 in IC_setting_list.txt (powerlaw or Sersic): 1
the third value of line4 in IC_setting_list.txt (potI, inertial frame potential:0; potR, rotating frame potential: 1): 0
the first value in line7 in user_settings_multi.txt (model name with number in count of values of various parameter): Ein_multicomp_spinL_axisLH0 Ein_multicomp_spinL_axisLH1
"is_modify_IC" in user_settings_multi.txt (spin): 0

