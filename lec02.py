import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import lzma
import imgsim
import cv2
import glob
import os 
import time

red = np.array([255,0,0])
yellow = np.array([255,255,0])
blue = np.array([0,0,255])

def hex_to_rgb(color):
    color_rgb = color.lstrip('#')
    color_rgb = np.array(list(int(color_rgb[i:i+2], 16) for i in (0, 2, 4)))
    return color_rgb

def rgb_to_hex(rgb):
    return '#'+ '%02x%02x%02x' % rgb

def euclid(color):
    disto_red = np.linalg.norm(color - red)
    disto_yellow = np.linalg.norm(color - yellow)
    disto_blue = np.linalg.norm(color - blue)
    min_value = np.array([disto_red, disto_yellow, disto_blue]).min()
    if disto_red == min_value:
        return rgb_to_hex(tuple(red))
    elif disto_yellow == min_value:
        return rgb_to_hex(tuple(yellow))
    else:
        return rgb_to_hex(tuple(blue))

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cos_sim_color(color):
    cos_red = cos_sim(color, red)
    cos_yellow = cos_sim(color, yellow)
    cos_blue = cos_sim(color, blue)
    compare = [cos_red, cos_yellow, cos_blue]
    if cos_red == max(compare):
        return rgb_to_hex(tuple(red))
    elif cos_yellow == max(compare):
        return rgb_to_hex(tuple(yellow))
    else:
        return rgb_to_hex(tuple(blue))
    
st.title('専門コース演習II 2023年6月28日の課題')
st.subheader('課題1: オレンジが赤、黄、青のどの色と近いのかを導出するプログラムを完成すること。')
color = st.color_picker('色を選択してください', '#FFA500')
# st.write('The current color is', color)
new_color_bydis = euclid(hex_to_rgb(color))
new_color_bycos = cos_sim_color(hex_to_rgb(color))
st.color_picker('選択された色はユークリッド距離でこの色に近いです',new_color_bydis)
st.color_picker('選択された色はこサイン類似度でこの色に近いです',new_color_bycos)


def vectorize(img):
    vtr = imgsim.Vectorizer()
    v=vtr.vectorize(img)
    return v

st.subheader('課題3: AugNet*と呼ばれる画像を与えると768次元のベクトルに変換する手法がある。これをもちいて、画像データ群からベクトルデータベースを構築し、さらにGradioなどのWebアプリケーションフレームワークを用いて、画像を入力として与えると、ベクトルデータベース内からその画像と近しい画像を出力するアプリケーションを構築すること。')

uploaded_file = st.file_uploader("検証したい画像をアップロードしてください！", type=("png", "jpeg"))

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    target = vectorize(opencv_image)
    # st.write(np.shape(target))
    # st.write(os.getcwd())
    os.chdir("./")
    try:
        os.chdir("./images")
    except:
        st.write("Loading images")
    distance = []
    with st.spinner('ちょっと待ってください！211つの画像のあるデータベースからユークリッド距離が一番近い画像を出力します'):
        with st.empty():
            for i,file in enumerate(glob.glob("*.png")):
                distance.append(np.linalg.norm(target - vectorize(cv2.imread(file))))
                st.write("ファイル " + str(i+1) + " を読み込んでいます")
    
    distance = np.array(distance)
    img_index = np.argmin(distance)
    st.success("計算が終わりました！一番近い画像との距離は " + str(np.amin(distance)) + " となります")
    st.balloons()
    st.image(cv2.resize(cv2.imread(glob.glob("*.png")[img_index]), (300,300)), channels="BGR", caption="一番近い画像")
    

        

    