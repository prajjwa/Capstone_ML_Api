
#server
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cwd = os.getcwd()

# print(cwd)

os.chdir(cwd)
# os.chdir("C:/Users/prajj/OneDrive/Desktop/capstoneserver/model/Predicting-sensory-evaluation-of-spinach-freshness-using-machine-learning-model-and-digital-images/Attached_file")

loaded_model = joblib.load('./models_grid.pkl')

CLASS_NAMES = ["not very fresh", "fresh", "fresh", "very fresh"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    df_train=preprocess(image)
    df_train_unlabel = df_train.drop('label', axis=1)
    scaler = StandardScaler()
    scaler.fit(df_train_unlabel)
    scale_train =scaler.transform(df_train_unlabel)
    predicted_class=loaded_model.predict(scale_train)[0]

    return {CLASS_NAMES[predicted_class-1]}


#@title model code

#Setting directly
# os.chdir("C:/Users/prajj/OneDrive/Desktop/capstoneserver/model/Predicting-sensory-evaluation-of-spinach-freshness-using-machine-learning-model-and-digital-images/Attached_file")
def preprocess(img_orig):
  #Loading label
  df_label = pd.read_csv('./data/label_all.csv', names=['label'])
  # Vector
  label = df_label.values
  label_all = label.flatten()
  #All data set
  N_img =1
  #Colum_Feature
  col_color=[]
  col_color.extend(['gry', 'b', 'g', 'r', 'h', 's', 'v', 'l', 'a', 'bl',
                    'gry_std','b_std', 'g_std', 'r_std','h_std', 's_std', 'v_std','l_std','a_std','bl_std',
                    'gry_min','b_min','g_min','r_min','h_min','s_min','v_min', 'l_min','a_min','bl_min'])
  feature_matrix_color = np.zeros((N_img,len(col_color)))
  feature_color = pd.DataFrame(feature_matrix_color, columns = col_color)
  for i in range(N_img):
      img=img_orig
    #   filename = imgName
    #   img = imgName
      #img = cv2.imread(filename)
      img = cv2.resize(img, (int(np.round(img.shape[1]/5)), int(np.round(img.shape[0]/5))))
      gry =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #Otsu method
      ret, img_thresh = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((3,3),np.uint8)
      img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
      img_thresh = cv2.bitwise_not(img_thresh)
      img_thresh = cv2.erode(img_thresh,kernel,iterations = 1)
      height, width, color = img.shape 
      img_thresh3 = np.zeros((height, width, 3), dtype = "uint8")
      img_thresh3[:,:,0], img_thresh3[:,:,1], img_thresh3[:,:,2] = img_thresh/255, img_thresh/255, img_thresh/255
      #Remove Background 
      img_Hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      dst_gry = gry * (img_thresh/255)
      dst_bgr = img * img_thresh3
      dst_Hsv = img_Hsv * img_thresh3
      dst_Lab = img_Lab * img_thresh3
      
      #Color feature extraction: gry,bgr,hsv,lab mean,std,min
      feature_color['gry'][i] = dst_gry.T.flatten()[np.nonzero(dst_gry.T.flatten())].mean()
      feature_color['b'][i] = dst_bgr.T[0].flatten()[np.nonzero(dst_bgr.T[0].flatten())].mean()
      feature_color['g'][i] = dst_bgr.T[1].flatten()[np.nonzero(dst_bgr.T[1].flatten())].mean()
      feature_color['r'][i] = dst_bgr.T[2].flatten()[np.nonzero(dst_bgr.T[2].flatten())].mean()
      feature_color['h'][i] = dst_Hsv.T[0].flatten()[np.nonzero(dst_Hsv.T[0].flatten())].mean()
      feature_color['s'][i] = dst_Hsv.T[1].flatten()[np.nonzero(dst_Hsv.T[1].flatten())].mean()
      feature_color['v'][i] = dst_Hsv.T[2].flatten()[np.nonzero(dst_Hsv.T[2].flatten())].mean()
      feature_color['l'][i] = dst_Lab.T[0].flatten()[np.nonzero(dst_Lab.T[0].flatten())].mean()
      feature_color['a'][i] = dst_Lab.T[1].flatten()[np.nonzero(dst_Lab.T[1].flatten())].mean()
      feature_color['bl'][i] = dst_Lab.T[2].flatten()[np.nonzero(dst_Lab.T[2].flatten())].mean()
      feature_color['gry_std'][i] = dst_gry.T.flatten()[np.nonzero(dst_gry.T.flatten())].std()
      feature_color['b_std'][i] = dst_bgr.T[0].flatten()[np.nonzero(dst_bgr.T[0].flatten())].std()
      feature_color['g_std'][i] = dst_bgr.T[1].flatten()[np.nonzero(dst_bgr.T[1].flatten())].std()
      feature_color['r_std'][i] = dst_bgr.T[2].flatten()[np.nonzero(dst_bgr.T[2].flatten())].std()
      feature_color['h_std'][i] = dst_Hsv.T[0].flatten()[np.nonzero(dst_Hsv.T[0].flatten())].std()
      feature_color['s_std'][i] =  dst_Hsv.T[1].flatten()[np.nonzero(dst_Hsv.T[1].flatten())].std()
      feature_color['v_std'][i] = dst_Hsv.T[2].flatten()[np.nonzero(dst_Hsv.T[2].flatten())].std()
      feature_color['l_std'][i] = dst_Lab.T[0].flatten()[np.nonzero(dst_Lab.T[0].flatten())].std()
      feature_color['a_std'][i] = dst_Lab.T[1].flatten()[np.nonzero(dst_Lab.T[1].flatten())].std()
      feature_color['bl_std'][i] = dst_Lab.T[2].flatten()[np.nonzero(dst_Lab.T[2].flatten())].std()
      feature_color['gry_min'][i] = dst_gry.T.flatten()[np.nonzero(dst_gry.T.flatten())].min()
      feature_color['b_min'][i] = dst_bgr.T[0].flatten()[np.nonzero(dst_bgr.T[0].flatten())].min()
      feature_color['g_min'][i] = dst_bgr.T[1].flatten()[np.nonzero(dst_bgr.T[1].flatten())].min()
      feature_color['r_min'][i] = dst_bgr.T[2].flatten()[np.nonzero(dst_bgr.T[2].flatten())].min()
      feature_color['h_min'][i] = dst_Hsv.T[0].flatten()[np.nonzero(dst_Hsv.T[0].flatten())].min()
      feature_color['s_min'][i] = dst_Hsv.T[1].flatten()[np.nonzero(dst_Hsv.T[1].flatten())].min()
      feature_color['v_min'][i] = dst_Hsv.T[2].flatten()[np.nonzero(dst_Hsv.T[2].flatten())].min()
      feature_color['l_min'][i] = dst_Lab.T[0].flatten()[np.nonzero(dst_Lab.T[0].flatten())].min()
      feature_color['a_min'][i] = dst_Lab.T[1].flatten()[np.nonzero(dst_Lab.T[1].flatten())].min()
      feature_color['bl_min'][i] = dst_Lab.T[2].flatten()[np.nonzero(dst_Lab.T[2].flatten())].min()

  #label_color_fearue
  df_att = pd.concat([df_label[:N_img],feature_color],axis=1)
  #Save the label and color information
  df_att.to_csv('./data/color_features_dummy.csv')
  #Load the label and color information
  df = pd.read_csv('./data/color_features_dummy.csv', index_col=0)
  #Seperating the dataset as response variable and feature variabes
  X = df.drop(['label'],axis=1)
  y = df['label']
  #Preparing for local feature by Orb
  #Train and Test splitting of data 
  X_train=X
  y_train=y
  index_train = X_train.index
  #Preparing_for_local_feature_detection
  orb = cv2.ORB_create()
  features_orb = []
  features_orb_bgr = [[],[],[]]
  features_orb_Lab = [[], [], []]
  features_orb_hsv = [[], [], []]
  for i in index_train:
      img = img_orig
      img = cv2.resize(img, (int(np.round(img.shape[1]/5)), int(np.round(img.shape[0]/5))))
      gry =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, img_thresh = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((3,3),np.uint8)
      img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
      img_thresh = cv2.bitwise_not(img_thresh)
      img_thresh = cv2.erode(img_thresh,kernel,iterations = 1)
      height, width, color = img.shape         
      img_thresh3 = np.zeros((height, width, 3), dtype = "uint8")
      img_thresh3[:,:,0], img_thresh3[:,:,1], img_thresh3[:,:,2] = img_thresh/255, img_thresh/255, img_thresh/255
      img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      img_Hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      dst_bgr = img * img_thresh3
      dst_Lab = img_Lab * img_thresh3
      dst_Hsv = img_Hsv * img_thresh3
      try:
          features_orb.extend(orb.detectAndCompute(gry, None)[1])
      except:
          pass
      
      for z in range(3):
          try:
              features_orb_bgr[z].extend(orb.detectAndCompute(dst_bgr[:,:,z], None)[1])
          except:
              pass
          try:
              features_orb_Lab[z].extend(orb.detectAndCompute(dst_Lab[:,:,z], None)[1])
          except:
              pass    
          try:
              features_orb_hsv[z].extend(orb.detectAndCompute(dst_Hsv[:,:,z], None)[1])
          except:
              pass    
  #Making bag of features
  c_n = 20
  visual_words_orb_bgr = [[],[],[]]
  visual_words_orb_Lab = [[],[],[]]
  visual_words_orb_hsv = [[],[],[]]
  visual_words_orb = cluster.MiniBatchKMeans(n_clusters=c_n,random_state=25).fit(features_orb).cluster_centers_
  for z in range(3):
      visual_words_orb_bgr[z] = cluster.MiniBatchKMeans(n_clusters=c_n,random_state=25).fit(features_orb_bgr[z]).cluster_centers_
      visual_words_orb_Lab[z] = cluster.MiniBatchKMeans(n_clusters=c_n,random_state=25).fit(features_orb_Lab[z]).cluster_centers_
      visual_words_orb_hsv[z] = cluster.MiniBatchKMeans(n_clusters=c_n,random_state=25).fit(features_orb_hsv[z]).cluster_centers_

  #Training
  #Vector of visual word index for training data set
  Vector_orb_gry = np.zeros([len(index_train),c_n])
  Vector_orb_bgr = [np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n])]
  Vector_orb_Lab = [np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n])]
  Vector_orb_hsv = [np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n]),np.zeros([len(index_train),c_n])]
  for i, I  in zip(range(len(index_train)),index_train):
      img = img_orig
      img = cv2.resize(img, (int(np.round(img.shape[1]/5)), int(np.round(img.shape[0]/5))))
      gry =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, img_thresh = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((3,3),np.uint8)
      img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
      img_thresh = cv2.bitwise_not(img_thresh)
      kernel = np.ones((3,3),np.uint8)
      img_thresh = cv2.erode(img_thresh,kernel,iterations = 1)
      height, width, color = img.shape 
      dst_bgr = np.zeros((height, width, 3), dtype = "uint8")             
      img_thresh3 = np.zeros((height, width, 3), dtype = "uint8")
      img_thresh3[:,:,0], img_thresh3[:,:,1], img_thresh3[:,:,2] = img_thresh/255, img_thresh/255, img_thresh/255
      img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      img_Hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      dst_bgr = img * img_thresh3
      dst_Hsv = img_Hsv * img_thresh3
      dst_Lab = img_Lab * img_thresh3
      
      #orb gry
      features_orb = orb.detectAndCompute(gry, None)[1]
      vector_orb = np.zeros(c_n)
      try:
          for f in features_orb:
              vector_orb[((visual_words_orb - f)**2).sum(axis=1).argmin()] += 1
      except:
              pass
      Vector_orb_gry[i,:] = vector_orb
      #Orb color
      for z in range(3):
          #orb bgr
          features_orb = orb.detectAndCompute(dst_bgr[:,:,z], None)[1]
          vector_orb = np.zeros(c_n)
          try:
              for f in features_orb:
                  vector_orb[((visual_words_orb_bgr[z] - f)**2).sum(axis=1).argmin()] += 1
          except:
                  pass
          Vector_orb_bgr[z][i,:] = vector_orb
          
          #orb Lab
          features_orb = orb.detectAndCompute(dst_Lab[:,:,z], None)[1]
          vector_orb = np.zeros(c_n)
          try:
              for f in features_orb:
                  vector_orb[((visual_words_orb_Lab[z] - f)**2).sum(axis=1).argmin()] += 1
          except:
                  pass
          Vector_orb_Lab[z][i,:] = vector_orb
          
          #orb hsv
          features_orb = orb.detectAndCompute(dst_Hsv[:,:,z], None)[1]
          vector_orb = np.zeros(c_n)
          try:
              for f in features_orb:
                  vector_orb[((visual_words_orb_hsv[z] - f)**2).sum(axis=1).argmin()] += 1
          except:
                  pass
          Vector_orb_hsv[z][i,:] = vector_orb
      #############################################################
  #Colum names bag of features by Orb
  n_orb_gry, n_orb_l, n_orb_a, n_orb_bl, n_orb_h, n_orb_s, n_orb_v = ['orb_gry']*c_n, ['orb_l']*c_n, ['orb_a']*c_n, ['orb_bl']*c_n,['orb_h']*c_n, ['orb_s']*c_n, ['orb_v']*c_n
  n_orb_b, n_orb_g, n_orb_r = ['orb_b']*c_n, ['orb_g']*c_n, ['orb_r']*c_n

  number = list(range(1,c_n+1))
  number_list = list(map(str, number))
  orb_gry_list = [a + b for a, b in zip(n_orb_gry, number_list)]
  orb_b_list = [a + b for a, b in zip(n_orb_b, number_list)]
  orb_g_list = [a + b for a, b in zip(n_orb_g, number_list)]
  orb_r_list = [a + b for a, b in zip(n_orb_r, number_list)]
  orb_l_list = [a + b for a, b in zip(n_orb_l, number_list)]
  orb_a_list = [a + b for a, b in zip(n_orb_a, number_list)]
  orb_bl_list = [a + b for a, b in zip(n_orb_bl, number_list)]
  orb_h_list = [a + b for a, b in zip(n_orb_h, number_list)]
  orb_s_list = [a + b for a, b in zip(n_orb_s, number_list)]
  orb_v_list = [a + b for a, b in zip(n_orb_v, number_list)]
  feature_orb_gry = pd.DataFrame(Vector_orb_gry, columns = orb_gry_list)
  feature_orb_b = pd.DataFrame(Vector_orb_bgr[0], columns = orb_b_list)
  feature_orb_g = pd.DataFrame(Vector_orb_bgr[1], columns = orb_g_list)
  feature_orb_r = pd.DataFrame(Vector_orb_bgr[2], columns = orb_r_list)
  feature_orb_L = pd.DataFrame(Vector_orb_Lab[0], columns = orb_l_list)
  feature_orb_a = pd.DataFrame(Vector_orb_Lab[1], columns = orb_a_list)
  feature_orb_bl = pd.DataFrame(Vector_orb_Lab[2], columns = orb_bl_list)
  feature_orb_h = pd.DataFrame(Vector_orb_hsv[0], columns = orb_h_list)
  feature_orb_s = pd.DataFrame(Vector_orb_hsv[1], columns = orb_s_list)
  feature_orb_v = pd.DataFrame(Vector_orb_hsv[2], columns = orb_v_list)

  col=[]
  col.extend(orb_gry_list + orb_b_list + orb_g_list + orb_r_list + orb_l_list + orb_a_list + orb_bl_list + orb_h_list + orb_s_list + orb_v_list)
  df_train = pd.concat([pd.DataFrame(y_train.values, columns=['label']),pd.DataFrame(X_train.values, columns=df.columns[1:]),
                        feature_orb_gry, feature_orb_b, feature_orb_g, feature_orb_r, feature_orb_L, feature_orb_a, feature_orb_bl, feature_orb_h, feature_orb_s, feature_orb_v],axis=1)
  return df_train    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)