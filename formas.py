# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

imagem_original = cv.imread('videos//pessoas2.jpg')

imagem_resize = cv.resize(imagem_original,(600,800))

imagem_cinza = cv.cvtColor(imagem_resize,cv.COLOR_BGR2GRAY)


_,thresh = cv.threshold(imagem_cinza,240,255,cv.THRESH_OTSU)

imagem_sem_ruido = cv.medianBlur(thresh,7)

contornos,_ = cv.findContours(imagem_sem_ruido,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

h,w = imagem_resize.shape[:2]
c = 3

imagem_contorno = np.full((h,w,c),0,np.uint8)

seedPic = np.random.randint(1,1e3)

video = cv.VideoWriter('./Quadro_'+str(seedPic)+'.mp4',cv.VideoWriter.fourcc(*'mp4v'),15,(w,h))

def geradorDeCores():
    cont = 0;
    R,G,B = (0,0,0)
    while cont <= seedPic:
        B,G,R = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        cont+=1
    return B,G,R


for contorno in contornos:
    
    video.write(imagem_contorno)
    
    approx = cv.approxPolyDP(contorno,0.002*cv.arcLength(contorno,True),True)
    
    cv.drawContours(image=imagem_contorno,
                contours=[approx],
                contourIdx = -1,
                color=(geradorDeCores()),
                thickness=cv.FILLED
                )
    
    cv.waitKey(150)
    
# cv.imshow('Original',imagem_original)
# cv.imshow('Cinza',imagem_cinza)
# cv.imshow('Thresh',thresh)
# cv.imshow('Sem_Ruido',imagem_sem_ruido)
    cv.imshow('Contorno',imagem_contorno)   


# imgAssinatura = np.zeros((50,w,3),np.uint8)
# cv.putText(imgAssinatura,'4ever Young  : Heron TF Gomes',(140,25),cv.FONT_HERSHEY_SCRIPT_SIMPLEX ,1,(190,190,190),1,cv.LINE_AA )
# cv.putText(imgAssinatura,'14/06/2020',(495,45),cv.FONT_HERSHEY_SCRIPT_SIMPLEX ,0.5,(190,190,190),1,cv.LINE_AA )    
# cv.putText(imgAssinatura,'Seed:'+str(seedPic),(0,45),cv.FONT_HERSHEY_SCRIPT_SIMPLEX ,0.5,(190,190,190),1,cv.LINE_AA )    


# quadro = cv.vconcat([imagem_contorno,imgAssinatura])

cv.imwrite('Quadro_Seed_'+str(seedPic)+'.png',imagem_contorno);


    
# cv.imshow('Quadro: '+str(seedPic),quadro)
video.release()
cv.waitKey()
cv.destroyAllWindows()