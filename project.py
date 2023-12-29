import streamlit as st
from streamlit_cropper import st_cropper
import numpy as np 
import cv2
from PIL import Image
import pathlib
import torch
import helpers 
from joblib import dump, load #Use save model

import re


st.set_page_config(layout="wide")



def reset():
    st.session_state.selection = 'Choose a language'

st.title('IMAGE CLASSIFICATION FOR TOURIST DESTINATIONS IN VIETNAM')
col1, col2 = st.columns(2)
with col1:
    st.header('Image')


    st.subheader('Upload image')
    img_file = st.file_uploader(label='.', type=['png', 'jpg'])
    if img_file:
        
        img = Image.open(img_file)
        st.subheader('Option')
        option = st.selectbox('.', ( 'Original', 'Crop'))
        if option == 'Original':
            st.image(img)
        elif option == 'Crop':
            # Get a cropped image from the frontend
            img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            # Manipulate cropped image at will
            st.write("Preview")
            _ = img.thumbnail((150,150))
            st.image(img)


            
with col2:


    st.header('PREDICT RESULT')
    if img_file:
        

        check_des = 0
        check_list = 0 
        label = helpers.output(img)
        # label = helpers.label_map[y_pred]
        st.write(f'**{label}**')
        option_lst = ['Choose a language', 'Chinese','Vietnamese', 'Japanese', 'English']
        option = st.selectbox('Select the language of the description',option_lst, index=0,key='selection')
        

        if option == 'Choose a language':
            print("wait")
        else :
            st.subheader(f'Describe with {option} language')
            #generate desribe 
            print("label: ",label)
            place = helpers.get_full_place(label)
            print("place: ",place)
            des = helpers.generate_describle(place, option)

            st.write(des)
            
            
            #get hotel and restaurant 
            type_lst = ['Choose a service','hotel','restaurant']
            type_service = st.selectbox(
            'Select type of service',type_lst,index=0)
            print("TYPE SERVICE: ",type_service)

            if type_service == 'Choose a service':
                pass
            elif type_service in type_lst[1:]:
                col3, col4 = st.columns(2)
                
                lst = helpers.generate_service_info(type_service, place, option)
            
                addresses = []
                name_lst = []

                pattern_1 = re.compile(r'^\d+\.\s.*', re.MULTILINE)
                pattern_2 = re.compile(r'^\d+\. .*?\nAddress: .*?$', re.MULTILINE | re.DOTALL)
                if re.findall(pattern_2, lst) == []:
                    matches = re.findall(pattern_1,lst)
                    
                    for item in matches:
                        if item[0].isdigit():
                            split_row = item.split("-",1)
                            name_lst.append(split_row[0][3:-1])
                        
                            if type_service == 'restaurant':
                                addresses.append(",".join(split_row[1].split(",")[-2:]))
                            else:
                                if split_row[1][1:].find("Address") != -1: 
                                    addresses.append(split_row[1][10:])
                                else:
                                    addresses.append(split_row[1][1:])


                elif re.findall(pattern_2,lst) != []:
                    matches = re.findall(pattern_2,lst)

                    
                    for item in matches:
                        if item[0].isdigit():
                            split_row = item.split("\n")
                            
                            name_lst.append(split_row[0][3:])
                            if type_service == "restaurant":
                                addresses.append(",".join(split_row[1].split(",")[-2:]))

                            else:
                                addresses.append(split_row[1][9:])  
        
                with col3:
                    if type_service == "hotel":
                        st.write("**List Hotel**")
                    
                    elif type_service == "restaurant":
                        st.write("**List Restaurant**")
                    # print(hotels)
                    for item in name_lst:
                        st.write('-',item)
                with col4:
                    st.write("**Address**")
                    # print(type_service, addresses)
                    for add in addresses:
                        st.write('-',add)

                st.button("Reset", on_click=reset)

                
            


        
        

