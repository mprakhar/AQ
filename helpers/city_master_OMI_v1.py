# CITY functions pertaining to city pixel locations for OMI
city_loc = (0,0)

def C1_Agra():
    city_loc_f = ( 252,1033)
    return city_loc_f
def C2_Ahmedabad():
    city_loc_f = ( 269,1011)
    return city_loc_f
def C3_Allahabad():
    city_loc_f = ( 258,1048)
    return city_loc_f
def C4_Amritsar():
    city_loc_f = ( 234,1020)
    return city_loc_f
def C5_Chennai():
    city_loc_f = ( 308,1041)
    return city_loc_f
def C6_Firozabad():
    city_loc_f = ( 252,1034)
    return city_loc_f
def C7_Gwalior():
    city_loc_f = ( 256,1033)
    return city_loc_f
def C8_Jodhpur():
    city_loc_f = ( 255,1013)
    return city_loc_f
def C9_Kanpur():
    city_loc_f = ( 255,1042)
    return city_loc_f
def C10_Kolkata():
    city_loc_f = ( 270,1074)
    return city_loc_f
def C11_Lucknow():
    city_loc_f = ( 253,1044)
    return city_loc_f
def C12_Ludhiana():
    city_loc_f = ( 237,1024)
    return city_loc_f
def C13_Mumbai():
    city_loc_f = ( 284,1012)
    return city_loc_f
def C14_New_Delhi():
    city_loc_f = ( 246,1029)
    return city_loc_f
def C15_Patna():
    city_loc_f = ( 259,1061)
    return city_loc_f
def C16_Raipur():
    city_loc_f = ( 275,1047)
    return city_loc_f
def C17_Bangalore():
    city_loc_f = (309,1031)
    return city_loc_f
def C18_Hyderabad():
    city_loc_f = (291,1034)
    return city_loc_f
def C19_Jaipur():
    city_loc_f = (252,1024)
    return city_loc_f
def C20_Pune():
    city_loc_f = (286,1016)
    return city_loc_f


city_list = ['Agra', 'Ahmedabad' , 'Allahabad'  , 'Amritsar' , 'Chennai' , 'Firozabad' , 'Gwalior' , 'Jodhpur' , 'Kanpur' , 'Kolkata' , 'Lucknow' , 'Ludhiana' ,
             'Mumbai' , 'New_Delhi' , 'Patna' , 'Raipur', 'Bangalore', 'Hyderabad', 'Jaipur', 'Pune']
city_list_C = ['C1_Agra', 'C2_Ahmedabad' , 'C3_Allahabad'  , 'C4_Amritsar' , 'C5_Chennai' , 'C6_Firozabad' , 'C7_Gwalior' , 'C8_Jodhpur' , 'C9_Kanpur' , 'C10_Kolkata' , 'C11_Lucknow' , 'C12_Ludhiana' ,
                'C13_Mumbai' , 'C14_New_Delhi' , 'C15_Patna' , 'C16_Raipur', 'C17_Bangalore', 'C18_Hyderabad', 'C19_Jaipur', 'C20_Pune']


# map the inputs to the function blocks
options = { 1 : C1_Agra,
            2 : C2_Ahmedabad ,
            3 : C3_Allahabad ,
            4 : C4_Amritsar ,
            5 : C5_Chennai ,
            6 : C6_Firozabad ,
            7 : C7_Gwalior ,
            8 : C8_Jodhpur ,
            9 : C9_Kanpur ,
            10 : C10_Kolkata ,
            11 : C11_Lucknow ,
            12 : C12_Ludhiana ,
            13 : C13_Mumbai ,
            14 : C14_New_Delhi ,
            15 : C15_Patna ,
            16 : C16_Raipur ,
            17 : C17_Bangalore ,
            18 : C18_Hyderabad ,
            19 : C19_Jaipur ,
            20 : C20_Pune
            }

'''
A code system to be used later on

1 : AGR: C1_Agra ,
2 : AMD: C2_Ahmedabad ,
3 : ALD: C3_Allahabad ,
4 : AMR: C4_Amritsar ,
5 : CHN: C5_Chennai ,
6 : FRZ: C6_Firozabad ,
7 : GWL: C7_Gwalior ,
8 : JDP: C8_Jodhpur ,
9 : KNP: C9_Kanpur ,
10 : KOL: C10_Kolkata ,
11 : LKO: C11_Lucknow ,
12 : LDH: C12_Ludhiana ,
13 : MUM: C13_Mumbai ,
14 : NDL: C14_New_Delhi ,
15 : PTN: C15_Patna ,
16 : RPR: C16_Raipur ,
17 : BLR: C17_Bangalore,
18 : HYD: C18_Hyderabad ,
19 : JPR: C19_Jaipur ,
20 : PUN: C20_Pune

'''