# import pandas as pd
# import numpy as np
# import datetime
# import matplotlib.pyplot as plt

# =============================================================================
# lst = [{"C1": 1, "C2": 2},
#         {"C1": 5, "C2": 10, "C3": 20}]
# print(pd.DataFrame(lst, index = ["R1", "R2"]))
# dc = {"C1": ["1", "3"],
#         "C2": ["2","4"]}
# print( pd.DataFrame(dc, index = ["R1", "R2"]) )
# lst = [[52,32],[45,85]]
# print(pd.DataFrame(lst, index = list('pq'), columns = list('ab')))
# df = pd.DataFrame({'A': [10., 20.],'B': "text",'C': [2,60],'D': 3+9j})
# print(df)
# print(df.dtypes)
# print(df.info())
# print(df.info(verbose=False))
# print((df.index))   #index of rows
# print(df.columns)   # index of columns
# print(df.values)    # display values of df
# =============================================================================

# =============================================================================
# int_values = [1, 2, 3, 4, 5]
# text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
# float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
# df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
#                    "float_col": float_values}, index = ["R1", "R2", "R3", "R4", "R5"])
# print(df)
# # Writing Info of the DataFrame in new File
# import io
# buffer1 = io.StringIO()
# df.info(buf=buffer1)
# s = buffer1.getvalue()
# with open("df_info.txt", "w", encoding="utf-8") as f:
#     f.write(s)
# =============================================================================

# =============================================================================
# df = pd.DataFrame([[0.23,'f1'],[5.36,'f2']], index = list('pq'), columns = list('ab'))
# lst = ['f30','f50','f2','f1']
# df=df.rename(columns = {'a':'A'})
# print(df)
# new_clm = list(np.random.randint(0,5,2))
# df['c'] = new_clm
# print(df)
# df['A']=df['A'].astype(complex)
# print(df)
# print(df[df.isin(lst).any(axis=1)])
# =============================================================================

# =============================================================================
# df = pd.DataFrame([[11, 202],
#                     [33, 44]],
#                     index = list('AB'),
#                     columns = list('CD'))
# df.to_excel(r'D:\practice\test_file.xlsx', sheet_name = 'Sheet1')
# print(pd.read_excel('test_file.xlsx', 'Sheet1'))
# df = pd.read_table(r'D:\practice\chat.txt')
# print(df)
# print(df.head(5))
# =============================================================================

# =============================================================================
# df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
#                    'mask': ['red', 'purple'],
#                    'weapon': ['sai', 'bo staff']})
# df.to_csv(index=True)
# print(df)
# =============================================================================

# =============================================================================
# df = pd.DataFrame([[18,10,5,11,-2],
#                     [2,-2,9,-11,3],
#                     [-4,6,-19,2,1],
#                     [3,-14,1,-2,8],
#                     [-2,2,4,6,13]],
#                   index = list('pqrst'),
#                     columns = list('abcde'))
# new_df=df[df.apply(sum, axis=1)%2==0]
# new_df.to_excel(r'D:\practice\file_df.xlsx', sheet_name = 'Sheet1')
# print(new_df)
# df_temp = new_df.copy()
# df_temp['m'] = df_temp.apply(np.prod, axis = 1)
# print(df_temp)
# path = r"D:\practice\file_df.xlsx"
# writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
# new_df.to_excel(writer, sheet_name = 'Sheet1')
# df_temp.to_excel(writer, sheet_name = 'Sheet2')
# writer.save()
# =============================================================================

# =============================================================================
# s = pd.Series(
#     [
#         "this is a regular sentence",
#         "https://docs.python.org/3/tutorial/index.html",
#         np.nan
#     ]
# )
# df = s.str.split(' ', None, expand=False)
# df1 = s.str.rsplit('/',2)
# print(df1)
# =============================================================================

# =============================================================================
# df = pd.read_table(r'D:\practice\chat_original.txt') #reading raw data from txt file
# df = df.iloc[:,0].str.split('M:', 1, expand=True) #splitting it into 2
# # print(df)
# ts = df.iloc[:,0].copy() #copying the 1st part to ts
# # print(ts)
# ts = ts.reset_index(drop = True) #resetting index
# ts.columns = ['Timestamp'] #column name
# df = df.iloc[:,1].str.split(':', 1, expand=True) #splitting the 2nd part into 2
# df = df.reset_index(drop = True)
# df.columns = ['Name','Convo'] #column names
# # print(df)
# # print(ts)
# df['Name'][df['Name'].str.contains(r'12345 45555',na=False)] = ' Friend8' #if there is this number in this column it will change as Friend8
# df['Name'][df['Name'].str.contains(r'98765 12222',na=False)] = ' Friend9' #if there is this number in this column it will change as Friend9
# df = df[df['Name'].str.contains(r'\d{2} \d{5} \d{5}') == False] #if this regex gives value as false then show only that arrays rest removed example : None, nan, etc
# # print(df)
# df = df[df['Convo'].str.contains(r'image omitted') == False] #if this column contains image omitted then
# df = df[df['Convo'].str.contains(r'video omitted') == False]
# df = df[df['Convo'].str.contains(r'GIF omitted') == False]
# df = df.reset_index(drop = True)
# all_chat_list = []
# for i in range(len(df['Name'].drop_duplicates())):
#     temp = df['Convo'][df['Name'] == df['Name'].drop_duplicates().reset_index(drop = True)[i]] #comparing name and deleting duplicates of it for each line i
#     temp = temp.reset_index(drop = True)
#     for j in range(1,len(temp)):
#         temp[0] += ' ' + temp[j] #appending all the messages of each person
#     all_chat_list.append(temp[0]) #appending in main chat list
#     del temp
# print(pd.DataFrame(all_chat_list))  #converting that array to dataframe
# print(list(all_chat_list)[6]) #6th index chat
# fg,cnt = np.unique(list(all_chat_list)[2].split(' '),return_counts=True) #2nd index chat message and its count
# b,c = np.asarray((fg,cnt)) #converting tuple into array
# b,c = b[c[:].argsort()][::-1],c[c[:].argsort()][::-1] #sorting and reversing each array
# print(b[:3],c[:3]) #displaying each array
# ts = ts.truncate(before = 1, after = 2) #showing only wanted arrays
# print(ts.head(5)) #first 5 arrays
# ts1 = ts[:1] #slicing 1st row
# ts2 = ts[3:] #slicing 3rd to all row
# print(ts1)
# ts = ts1.append(ts2) #appending 1st and 2nd
# # print(ts)
# # print(ts.tail(5)) #last 5 rows
# ts = ts+'M'  #appending M to it
# pd.to_datetime(ts) #converting it to datetime
# print(ts.shape)
# print(ts.isnull())
# print(ts.count())
# ts = ts.reset_index(drop = True)
# print(ts)
# splitted_ts= ts.str.split(', ')
# print(splitted_ts)
# for i in range(len(ts)):
#     temp = datetime.datetime.strptime(splitted_ts[i][1], '%I:%M:%S %p') #assigning the corresponding values
#     splitted_ts[i][1] = datetime.datetime.strftime(temp, '%H') #converting it to hours
# print(splitted_ts)
# hrs = [splitted_ts[i][1] for i in range(len(splitted_ts))] #getting all hours values
# hr,occ = np.unique(hrs,return_counts=True) #getting all hours along with count
# print(hr)
# print(occ)
# plt.plot(hr, occ)
# plt.grid('on')
# plt.xlabel('24 Hours')
# plt.ylabel('Frequency')
# plt.title('Frequent chat timings')
# =============================================================================

# =============================================================================
# cols_name = ['IDENTIF','RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
# df = pd.read_csv(r'D:\practice\data1.csv', names= cols_name, na_values = '?')
# print(df)
# print(df.isna().sum()) #sum the NA values with column names
# temp_df = df.columns[df.columns.str.contains("[NHS]$")] #return index of columns(column name) ending with N H S letter
# print(temp_df)
# df1 = df[temp_df].dropna(axis=1, thresh=100) #drop columns which has atleast 100 non-na values. Here inplace = False by default so it wont do operations in the existing df
# print(df1)
# df2 = df.drop(columns=temp_df) #dropping all columns ended with with N H S letter
# print(df2)
# df2[df1.columns] = df1[df1.columns] #inserting that column in df2 dataframe
# print(df2)
# df2.info()
# df2.dropna(axis=0, thresh=8, inplace=True) #droppping rows which has atleast 8 non-na values. Here inplace = True because it does operations and return the updated dataframe
# df2.reset_index(drop=True, inplace=True)
# print(df2)
# print(df2.shape)
# cols = df2.columns[df2.isnull().any()] #any columns having any number of null values
# print(cols)
# df2[cols]=df2[cols].fillna(df2.mode().iloc[0]) #in that column which has NA values fill those with mode(repeated many number of time) values
# print(df2.isnull().sum()) #summing all the null values in each columns
# =============================================================================

# df = pd.DataFrame({'A' : [1,-4,7],
#                     'B' : [-2,5,-8],
#                     'C' : [-3,-6,9]},
#                     index = list('STU'))
# print(df.apply(np.sum, axis = 0).sum())

# print(None == None)
# print(np.nan == np.NaN)
# print(np.nan == np.nan)

# df = pd.DataFrame([[2,5],
#                    [56,20]],
#                    index = list('pq'),
#                     columns = list('ab'))

# def loc(a,b):
#     try:
#         out = df.loc[a,b]
#     except:
#         return 0
#     return out

# def iloc(a,b):
#     try:
#         out = df.iloc[a,b]
#     except:
#         return 0
#     return out

# print( iloc(0,'b') )
# print( loc(0,'b') )
# print( loc('q','a') )
# print( iloc('q','b') )

# df1 = pd.DataFrame({'A': [1,4,7],
#                     'B': [2,5,8],
#                     'C': [3,6,9]},
#                     index = list ('PQR'))
# df2 = pd.DataFrame({'A': [1,-4,-7],
#                     'B': [-2,5,-8],
#                     'C': [-3,-6,9]},
#                     index = list ('STU'))
# print(df1.sub(df2))
# print(df1.sub(df2).count().sum())

# df = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                    [3,4,np.nan,-1],
#                    [np.nan, np.nan, np.nan, 5]],
#                   columns = list('ABCD'))
# print(df.dropna(axis = 1, how = 'all'))

# df = pd.DataFrame([[15,12],
#                   [33,54],
#                   [10,32]],
#                   index = ['one','two','three'],
#                   columns = ['col1', 'col2'])
# print(df.filter(regex = 'e$', axis = 0))

# df = pd.DataFrame([[15,12],
#                    [33,54],
#                    [10,32]],
#                    columns = list('AB'))
# print(df.eval('B-A').median())

# data1 = pd.DataFrame([[15, 12, -3],
#                     [33, 54, 21],
#                     [10, 32, 22]],
#                     columns = list('ABC'))
# data2 = pd.DataFrame([[10, 1, 3],
#                     [33, -54, 2],
#                     [10, 0.32, 2]],
#                     columns = list('DEF'))
# print(data1)
# print(data2)
# print(pd.concat([data1,data2], axis = 1))

# =============================================================================
# sys = ['s1','s1','s1','s1',
#         's2','s2','s2','s2']
# net_day = ['d1','d1','d2','d2',
#         'd1','d1','d2','d2']
# spd = [1.3, 11.4, 5.6, 12.3,
#         6.2, 1.1, 20.0, 8.8]
# df = pd.DataFrame({'set_name':sys,
#                     'spd_per_day':net_day,
#                     'speed':spd})
# new_df = df.groupby(["set_name","spd_per_day"],as_index = False).median()
# print(new_df)
# new_df.columns = pd.MultiIndex.from_tuples([("set_name", ""), ("spd_per_day", ""), ("speed", "median")])
# new_df = new_df.sort_values(("speed","median"),ascending=True)
# print(new_df)
# =============================================================================






























