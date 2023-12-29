from secret_key import openapi_key,serper_api_key, google_api_key,google_cse_id
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from langchain.agents import AgentType, initialize_agent, load_tools
import os
import pickle
import cv2
from PIL import Image
import numpy as np
from skimage import feature
import tensorflow as tf
import time
import streamlit as st
from langchain.document_transformers import DoctranTextTranslator
from langchain.schema import Document
from langchain.output_parsers import NumberedListOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate, AIMessagePromptTemplate,HumanMessagePromptTemplate

os.environ['OPENAI_API_KEY'] = openapi_key
os.environ['SERPER_API_KEY'] = serper_api_key
os.environ['GOOGLE_API_KEY'] = google_api_key
os.environ['GOOGLE_CSE_ID'] = google_cse_id

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet', 
    pooling='avg'
)
pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(20, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.load_weights('pretrained/MB_Model.h5')

def get_full_place(place):
    destination = {
    "thành phố Hồ Chí Minh":["Dinh Độc Lập","Chợ Bến thành","Nhà thờ Đức Bà","Bưu điện trung tâm"] ,
    "tỉnh Bà Rịa Vũng Tàu": ["Tượng Chúa Giang Tay Vũng Tàu"],
    "tỉnh Phú Yên": ['Gành đá đĩa'],
    "thành phố Đà Nẵng": ["Cầu vàng bà nà hills"],
    "thành phố Hà Nội": ["Lăng Bác Hồ","Hồ Gươm","Nhà Thờ Lớn"],
    "tỉnh Thừa Thiên-Huế": ["Kinh Thành Huế"],
    "tỉnh Quảng Ninh": ["Vịnh Hạ Long"],
    "tỉnh Quảng Nam": ["Thánh Địa Mỹ Sơn","Phố Cổ Hội An"],
    "tỉnh Hà Giang": ["Cột cờ Lũng Cú"],
    "tỉnh Quảng Bình": ["Động Phong nha kẻ bàng"],
    "tỉnh Cao Bằng": ["thác Bản Giốc"],
    "tỉnh Lâm Đồng": ["Ga Đà Lạt"],
    "thành phố Cần Thơ": ["Chợ Nổi"],
    "tỉnh Ninh Bình": ["Tràng An Ninh Bình"]
    }
    temp = [key for key, value in destination.items() if place in value][0]
    return place + ", " + temp


def prompt(task,lang,place, option = None):

    province = place.split(",")[1][1:]
    print(province)
    if task == "des":
        sys_template = 'Generate a short description approximately 100 words of a destination in Vietnam translated in {lang} language.'
    elif task == "list": 
        if option == "restaurant":
            sys_template = "Please provide a list of 5 {option}s located in a destination, Vietnam, along with their respective addresses."
        else:
            sys_template = "Please provide a list of 5 {option}s located near a destination, Vietnam, along with their respective addresses."


    

    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    
    if task == "des":
        result = chat(chat_prompt.format_prompt(lang = lang,text = place).to_messages()).content

    elif task == "list": 
        if option == "hotel":
            result = chat(chat_prompt.format_prompt( option = option,text = place).to_messages()).content

        else:
            result = chat(chat_prompt.format_prompt( option = option,text = province).to_messages()).content
        
    
    

        
    print("result: ",result)
    return result
    
@st.cache_data
def generate_describle(place, language):

    result = prompt("des",language,place)
    return result


@st.cache_data
def generate_service_info(option, name, language):
    if option == 'hotel':
        result = prompt("list",language,name,option)

    elif option == 'restaurant':
        result = prompt("list",language,name,option)
        

    print("option ", option)
        
    print(result)
    return result
def TinhHog(img):
  (hog, hog_image) = feature.hog(img, orientations=9,pixels_per_cell=(6, 6), cells_per_block=(3, 3),block_norm='L2-Hys', visualize=True, transform_sqrt=True)

  return hog

def predict_y(img):
    img_arr = np.array(img)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    img1 = cv2.resize(img_gray, (100, 100))
    hog = TinhHog(img1)
    hog = hog.reshape(1,-1)
    y = model.predict(hog)

    return int(y[0])

def output(img):
    # img=load_img(location,target_size=(224,224,3))
    img_arr = np.array(img)
    img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img1 = cv2.resize(img, (224,224))
    img=img_to_array(img1)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    label = map_to_id[y_class[0]]
    place = label_map[label]

    return place



map_to_id = {
    0 : 0,
    1 : 1,
    2 : 10,
    3 : 11,
    4 : 12,
    5 : 13,
    6 : 14,
    7 : 15,
    8 : 16,
    9 : 17,
    10: 18,
    11: 19,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    18: 8,
    19: 9
}

label_map = {
    0 : 'Lăng Bác Hồ',
    1 : 'Tràng An Ninh Bình',
    2 : 'Kinh Thành Huế',
    3 : 'Nhà thờ Đức Bà',
    4 : 'Vịnh Hạ Long',
    5 : 'Dinh Độc Lập',
    6 : 'Thánh Địa Mỹ Sơn',
    7 : 'Hồ Gươm',
    8 : 'Bưu điện trung tâm',
    9 : 'Tượng Chúa Giang Tay Vũng Tàu',
    10: 'Cầu vàng bà nà hills',
    11: 'Cột cờ Lũng Cú',
    12: 'Động Phong nha kẻ bàng',
    13: 'Phố Cổ Hội An',
    14: 'thác Bản Giốc',
    15: 'Nhà Thờ Lớn',
    16: 'Chợ Bến thành',
    17: 'Ga Đà Lạt',
    18: 'Chợ Nổi',
    19: 'Gành đá đĩa'
}
