import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Bar graph basics
# =============================================================================
# a = pd.read_csv('pokemon_alopez247.csv') #it will read csv files and store it in a variable
# b = np.unique(a.iloc[:,2],return_counts=True) #it will return 2 arrays 0th row contains names of the group and 1st row contains the count of the each unique group
# total_dgrps = len(b[0]) #length of the unique group names
# dgrps_name = b[0] #unique group names
# dgroups_cnt = b[1] #count of the unique group names
# type_1_grps = np.arange(total_dgrps) #gives the numbers from 0 to 18 as length of the unique group is 18 with step count as 1 by default
# plt.rcParams['figure.figsize'] = [15, 5] #changes the width and height of a plot because name in x axis are overlapped
# plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"] #reset the plot size
# plt.bar(type_1_grps, dgroups_cnt, width=0.5,align='edge',
#                   alpha = 0.5,   # tranparency factor
#                   color = 'green',   # color factor
#                   label='Pokemon count respective to their Type_1') #1st parameter: xaxis-names or numbers, 2nd parameter:height-height of the bars, 3rd param:
#     #width-default value is 0.8, 4th param: align-center or edge default is center
# plt.legend(loc='best') #label for that graph will be upper left
# plt.xticks(type_1_grps + 0.5/2, dgrps_name) #position of the xaxis name
# # plt.xticks(rotation=45) #rotation of the xaxis name
# #OR
# a = pd.read_csv('pokemon_alopez247.csv') #it will read csv files and store it in a variable
# b = np.unique(a.iloc[:,2],return_counts=True) #it will return 2 arrays 0th row contains names of the group and 1st row contains the count of the each unique group
# dgrps_name = b[0] #unique group names
# dgroups_cnt = b[1] #count of the unique group names
# plt.rcParams['figure.figsize'] = [15, 5] #changes the width and height of a plot because name in x axis are overlapped
# plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"] #reset the plot size
# plt.bar(dgrps_name, dgroups_cnt, width=0.5,
#                  alpha = 0.5,   # tranparency factor
#                  color = 'green',   # color factor
#                  label='Pokemon count respective to their Type_1') #1st parameter: xaxis-names or numbers, 2nd parameter:height-height of the bars, 3rd param:
#     #width-default value is 0.8, rest parameters used here are understandable
# plt.legend(loc='best') #label for that graph will be upper left
# =============================================================================

#Bar graphs with error plots
# =============================================================================
# n_grps = np.arange(5)
# bar_width_men = -0.4
# bar_width = 0.4
# men_sc = [20, 30, 10, 50, 90]
# err_men_sc = [2, 3, 4, 5, 4]
# women = [10, 123, 19, 60, 40]
# err_women_sc = [1, 6, 2, 8, 7]
# x_name = ['A','B','C','D','E']
# plt.bar(n_grps, men_sc, bar_width_men, color='red', align='edge', yerr=err_men_sc, ecolor='black', capsize=5)
# plt.bar(n_grps, women, bar_width, color='green', align='edge', yerr=err_women_sc, ecolor='black', capsize=5)
# plt.legend(['men','women'],loc='upper right')
# plt.xticks(n_grps, x_name)
# #OR
# n_grps = np.arange(5)
# bar_width = -0.4
# men_sc = [20, 30, 10, 50, 90]
# err_men_sc = [2, 3, 4, 5, 4]
# women = [10, 123, 19, 60, 40]
# err_women_sc = [1, 6, 2, 8, 7]
# x_name = ['A','B','C','D','E']
# plt.bar(n_grps, men_sc, bar_width, color='red', align='edge', yerr=err_men_sc, ecolor='black', capsize=5)
# plt.bar(n_grps-bar_width, women, bar_width, color='green', align='edge', yerr=err_women_sc, ecolor='black', capsize=5)
# plt.legend(['men','women'],loc='upper right')
# plt.xticks(n_grps, x_name)
# =============================================================================

#Advanced bar graph
# =============================================================================
# a = pd.read_csv('pokemon_alopez247.csv') #it will read csv files and store it in a variable
# b = np.unique(a.iloc[:,2],return_counts=True) #it will return 2 arrays 0th row contains names of the group and 1st row contains the count of the each unique group
# total_dgrps = len(b[0]) #length of the unique group names
# dgrps_name = b[0] #unique group names
# dgroups_cnt = b[1] #count of the unique group names
# type_1_grps = np.arange(total_dgrps) #gives the numbers from 0 to 18 as length of the unique group is 18 with step count as 1 by default
# plt.rcParams['figure.figsize'] = [15, 5] #changes the width and height of a plot
# # plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"] #reset the plot size
# grap = plt.bar(type_1_grps, dgroups_cnt, width=0.5,align='edge',
#                   alpha = 0.5,   # tranparency factor
#                   color = np.random.rand(18,4))
# plt.legend(grap, dgrps_name, bbox_to_anchor=(1.128,1.015)) #label for that graph will be upper left
# plt.xticks(type_1_grps + 0.5/2, dgrps_name) #position of the xaxis name
# # plt.xticks(rotation=45) #rotation of the xaxis name
# plt.xlabel('Type')
# plt.ylabel('Pokemon count')
# plt.title('Number of Pokemon per Type_1')
# plt.grid()
# plt.ylim(0,130)
# =============================================================================

# Pie chart with subplots
# =============================================================================
# # df = pd.read_csv('pokemon_alopez247.csv')
# # df_pie = df[['Type_1', 'Attack', 'Defense', 'Speed', 'HP']].copy()
# # df_new1, df_new2 = np.unique(df_pie.iloc[:,0],return_counts=True)
# # df_index = np.argsort(df_new2)
# # df_new1 = np.array(df_new1[df_index][::-1]).reshape(18,1)
# # df_new2 = np.array(df_new2[df_index][::-1]).reshape(18,1)
# # df_new = np.concatenate((df_new1,df_new2),axis=1)
# # print(df_new[:4])
# # OR
# # df = pd.read_csv('pokemon_alopez247.csv')
# # df_pie = df[['Type_1', 'Attack', 'Defense', 'Speed', 'HP']].copy()
# # df_slic = np.unique(df_pie.iloc[:,0],return_counts=True)
# # df_new = pd.DataFrame(df_slic).T
# # df_new=df_new.rename(columns = {0:'type',1:'count'})
# # df_new1 = df_new.sort_values(by=['count'], axis=0, ascending=False, inplace=False).reset_index(drop=True).iloc[:4,:]
# # OR
# df = pd.read_csv('pokemon_alopez247.csv')
# df_pie = df[['Type_1', 'Attack', 'Defense', 'Speed', 'HP']].copy()
# df_slic = np.unique(df_pie.iloc[:,0],return_counts=True)
# df_new = np.stack(df_slic,axis=1)
# df_arr = np.asarray(df_new)
# df_arr1 = df_arr[np.argsort(df_arr[:,1])][::-1][:4]
#
# plt.rcParams['figure.figsize'] = [10, 5]
# df_pie1 = df_pie.loc[df_pie.loc[:,'Type_1'].str.contains(r'(Water|Normal|Grass|Bug)')]
# type_1_names = np.sort(df_arr1[:,0])
# df_grp = df_pie1.groupby('Type_1').mean()
#
# names = df_grp.columns
# col = ['gold', 'lightcoral', 'yellowgreen', 'lightskyblue']
# exp = (0, 0, 0, 0.1) # takes out only the 4th slice
# fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
# ax = [ax1, ax2, ax3, ax4]
# for i in range(0,4):
#     values = df_grp.iloc[i,:]
#     ax[i].pie(values, explode = exp,
#             labels = names, colors = col,
#             autopct='%1.1f%%',   # display value can also be .2f
#             shadow=True,
#             startangle=90)
#     ax[i].set_aspect('equal')
#     ax[i].set_title(type_1_names[i])
# plt.suptitle('Comparing major features of 4 most frequent Pokemon Group',
#              fontsize = 14,
#              fontweight = 'bold')
# =============================================================================

# Animated graph
# =============================================================================
# from matplotlib import animation
# n = 30
# x = np.arange(0,1,0.001)
# y = np.ones( (1000,n) )
# for i in range(0,n):
#     y[:,i] = np.sin(2 * np.pi * x) * i+1
# def func(arg):
#      plt.cla()   #Clear axis
#      plt.plot(y[:,arg])
#      plt.ylim(-30,30)
# fig = plt.figure(figsize=(5,4))
# to_save = animation.FuncAnimation(fig, func, frames=30)
# plt.show()
# =============================================================================
# Scatter plot
# =============================================================================
# df = pd.read_csv('pokemon_alopez247.csv')
# tot_power = df.iloc[:,4]
# print(tot_power.head(4))
# catch_rate = df.iloc[:,21]
# print(catch_rate.head(4))
# fig, ax = plt.subplots()
# p = ax.scatter(catch_rate, tot_power, c = 'g')
# ax.grid()
# ax.set_xlabel('Catch Rate')
# ax.set_ylabel('Total Power')
# ax.set_title('Pokemon Catch Rate vs their Power')
# plt.legend([p],['Pokemons'])
#
# import matplotlib.patches as patches
# import matplotlib.transforms as transforms
# trans = transforms.blended_transform_factory(
#     ax.transData,ax.transAxes)
# rect = patches.Rectangle((44,0), width=2, height=1,
#                          transform=trans, color='red',
#                          alpha=0.4)
# ax.add_patch(rect)
#
# catch_rate_45 = df[df.loc[:,'Catch_Rate'] == 45]
# pow_330 = catch_rate_45[catch_rate_45.loc[:,'Total'] <= 330]
# print("Number of such Pokemons:", len(pow_330))
# # Top 10 adamant Pokets
# print(pow_330.loc[:,'Name'].head(10))
# =============================================================================

# 3D Scatter plot
# =============================================================================
# from mpl_toolkits import mplot3d
# x, y, z = np.random.rand(3,50)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# c = ['r','y','b'] * len(x)
# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_zlabel('z-axis')
# for x,y,z,c in zip(x,y,z,c):
#   ax.scatter(x,y,z,c=c,alpha=0.5,marker='o')
# #considering you saved your Axes under variable ax
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
# plt.show()
# # Or
# # from mpl_toolkits.mplot3d import Axes3D
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # x, y, z = np.random.rand(3,50)
# # c = x + y + z
# # ax.scatter(x, y, z, c=c,alpha=0.5,marker='o')
# # ax.set_title('3d Scatter plot')
# # ax.set_xlabel('X Label')
# # ax.set_ylabel('Y Label')
# # ax.set_zlabel('Z Label')
# # # consiering you saved your Axes under variable ax
# # for angle in range(0, 360):
# #     ax.view_init(30, angle)
# #     plt.draw()
# #     plt.pause(.001)
# # plt.show()
# =============================================================================

# Line graph
# =============================================================================
# df = pd.read_csv('pokemon_alopez247.csv')
# bulbasaur = list(df.iloc[0,4:11])
# charmander = list(df.iloc[3,4:11])
# squirtle = list(df.iloc[6,4:11])
# pokets = [bulbasaur, charmander, squirtle]
# # Converting height from meters to inches
# # Weight remains in kg
# bul_hw = [df.iloc[0,19]*3.281, df.iloc[0,20]]
# char_hw = [df.iloc[3,19]*3.281, df.iloc[3,20]]
# squ_hw =  [df.iloc[6,19]*3.281, df.iloc[6,20]]
# hw = [bul_hw, char_hw, squ_hw]
#
# fig = plt.figure(0)
# # Line plot
# ax1 = plt.subplot2grid((4,3), (0,0), colspan=3,rowspan=2)
# ax1.plot(bulbasaur,'g-',bulbasaur,'go')
# ax1.plot(charmander,'r-',charmander,'ro')
# ax1.plot(squirtle,'b-',squirtle,'bo')
# ax1.set_xlim(-0.3,6.3)
#
# text_x_coord = [0.08, .19, 0.34, 0.49, 0.64, 0.795, 0.945]
# text_y_coord = [0.95, 0.3, 0.65, 0.75, 0.75, 0.75, 0.72]
# rot = [0, 45, 45, 45, 45, 45, 45]
# txt = [r'$T_{OTAL}$', r'$HP$', r'$A_{TTACK}$', r'$D_{EFENSE}$',
#         r'$S_{P}\_A_{TK}$',r'$S_{P}\_D_{EF}$',r'$S_{PEED}$']
# for i in range(0, 7):
#     ax1.text(text_x_coord[i], text_y_coord[i], txt[i],
#               transform = ax1.transAxes, # makes width and height in percentage
#               va = 'top',
#               rotation = rot[i],
#               bbox = dict(
#                       boxstyle = 'round',
#                       facecolor = 'wheat',
#                       alpha = 0.78))  # alpha -> transparency
# =============================================================================

# subplot2grid funtion in Line graph
# =============================================================================
# fig = plt.figure()
# axes1 = plt.subplot2grid((4, 4), (0, 0),
#                          colspan = 4)
# axes2 = plt.subplot2grid((4, 4), (1, 0),
#                           colspan = 3)
# axes3 = plt.subplot2grid((4, 4), (1, 2),
#                           rowspan = 3)
# axes4 = plt.subplot2grid((4, 4), (2, 0))
# axes5 = plt.subplot2grid((4, 4), (2, 1))
# fig.tight_layout()
# =============================================================================

# Bar Chart with annotate function
# =============================================================================
# df = pd.read_csv('pokemon_alopez247.csv')
# bulbasaur = list(df.iloc[0,4:11])
# charmander = list(df.iloc[3,4:11])
# squirtle = list(df.iloc[6,4:11])
# pokets = [bulbasaur, charmander, squirtle]
# # continuing from previous code
# ax2 = plt.subplot2grid((4,3), (1,0), colspan=1)
# ax3 = plt.subplot2grid((4,3), (2,0), colspan=1)
# ax4 = plt.subplot2grid((4,3), (3,0), colspan=1)
# ax = [ax2, ax3, ax4]
# colors = ['g', 'r', 'b']
# for i in range(0, 3):
#     bp = ax[i].boxplot(pokets[i][1:],patch_artist=True)
#     # Adding colors to edges
#     for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#             plt.setp(bp[element], color='k')
#     # Adding color inside the box
#     for patch in bp['boxes']:
#         patch.set(facecolor=colors[i])
#     ax[i].set_ylim(35,70)
# ax3.annotate('Median', xy=(1.09, 51),
#               xytext=(1.2, 60),
#             arrowprops=dict(facecolor='wheat',
#                             shrink=0.001),)
# =============================================================================

# Subplot varities to understand add_subplot
# =============================================================================
# # fig = plt.figure()
# # fig.add_subplot(221)   #top left
# # fig.add_subplot(222)   #top right
# # fig.add_subplot(223)   #bottom left
# # fig.add_subplot(224)   #bottom right
# # plt.show()
#
# # OR
#
# # fig = plt.figure()
# # fig.add_subplot(1, 2, 1)   #top and bottom left
# # fig.add_subplot(2, 2, 2)   #top right
# # fig.add_subplot(2, 2, 4)   #bottom right
# # plt.show()
#
# # OR
#
# # plt.figure(figsize=(8,8))
# # plt.subplot(3,2,1)
# # plt.subplot(3,2,3)
# # plt.subplot(3,2,5)
# # plt.subplot(2,2,2)
# # plt.subplot(2,2,4)
# =============================================================================

# Exercise has text, annotate
# =============================================================================
# x = np.arange(0.0, 5.0, 0.1)
# y = np.cos(2 * np.pi * x) * np.exp(-x)
# plt.figure(figsize=(9,5))
# plt.plot(x,y, marker='o', markerfacecolor='darkturquoise', alpha=0.75, markersize=8, color='k')
# plt.grid()
# plt.xlabel("Abscissa")
# plt.ylabel("Ordinate")
# plt.text(1,0.4,(r"$\theta=60^\circ$"))
# plt.title("Figure Title")
# plt.suptitle('Main Heading',fontsize=14, fontweight='bold')
# plt.annotate('2nd Crest', xy=(2, 0.2), xytext=(2.5, 0.3), arrowprops=dict(facecolor='green', shrink=0.05))
# =============================================================================

# Normal Exercise of text position
# =============================================================================
# x = np.array([1,1,1,1,2,3,4,4,4,4,3,2,1])
# y = np.array([1,2,3,4,4,4,4,3,2,1,1,1,1])
# ax = plt.subplots()
# plt.text(3.7, 1.1, "Text1")
# plt.text(3.9, 2.25, "Text2", rotation=90)
# plt.text(1.0, 3.5, "Text3\nNextLine", rotation=45)
# plt.axis([1.0, 4.0, 1.0, 4.0])
# =============================================================================

# Horizontal bar chart
# =============================================================================
# df = pd.read_csv('pokemon_alopez247.csv')
# colors = ['g', 'r', 'b']
# ax5 = plt.subplot2grid((4,3), (1,1), colspan=2, rowspan = 3)
# bar_width = 0.2
# wdt = 0
# names = ['Bulbasaur','Charmander','Squirtle']
# bul_hw = [df.iloc[0,19]*3.281, df.iloc[0,20]]
# char_hw = [df.iloc[3,19]*3.281, df.iloc[3,20]]
# squ_hw =  [df.iloc[6,19]*3.281, df.iloc[6,20]]
# hw = [bul_hw, char_hw, squ_hw]
# for i in range(0, 3):
#     p=ax5.barh(np.arange(2) + wdt, hw[i],bar_width,
#              alpha = 0.5,
#              color = colors[i],
#              label = names[i])
#     wdt += bar_width
# ax5.legend(loc = 'lower right')
# ax5.set_xlim(0,12)
# ax5.set_ylim(-0.2,1.8)
# plt.yticks(np.arange(2) + (bar_width)*1.5,
#            ('Height', 'Weight'))
# plt.tight_layout()
# plt.style.use('seaborn-dark')
# =============================================================================



























