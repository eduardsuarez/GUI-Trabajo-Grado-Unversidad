from tkinter import *
from tkinter import filedialog

import cv2
import imutils
import joblib
import numpy as np
import winsound
from PIL import Image
from PIL import ImageTk
from keras.models import load_model




def elegirImagen():
    global imagen
    global knn
    global Clases
    imag1_prediccion = []
    global pathImagen
    pathImagen = filedialog.askopenfilename(filetypes=[
        ("imagen", ".jpg"),
        ("imagen", ".jpeg"),
        ("imagen", ".png")])
    if len(pathImagen) > 0:
        # leer la imagen de entrada
        imagen = cv2.imread(pathImagen)
        try:
            eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
            image_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            ojos = eyeCascade.detectMultiScale(image_gray, 1.1, 4)

            for x, y, w, h in ojos:
                roi_gray = image_gray[y:y + h, x:x + w]
                roi_color = imagen[y:y + h, x:x + w]
                ojo = eyeCascade.detectMultiScale(roi_gray)
                if len(ojo) == 0:
                    print("ojo no detectado")

                else:
                    for (ox, oy, ow, oh) in ojo:
                        ojos_roi = roi_color[oy:oy + oh, ox:ox + ow]

                imagenFinal = cv2.cvtColor(ojos_roi, cv2.COLOR_BGR2GRAY)
                imagenFinal = cv2.resize(imagenFinal, (64, 64))

            imagenMos = cv2.resize(imagenFinal, (100, 100))
            # aplanar la imagen
            image_array = np.array(imagenFinal)

            pixeles = image_array.flatten()

            # agregar la imágen aplanada a
            imag1_prediccion.append(pixeles)

            # visualizar la imagen de entrada en la GUI
            im = Image.fromarray(imagenMos)
            img = ImageTk.PhotoImage(image=im)
            imagenEntrada.configure(image=img)
            imagenEntrada.imagen = img

            # etiqueta de imagen de entrada
            lblImg1 = Label(top, text=" Imagen Entrada", bg="#69a3df", relief="flat", cursor="hand2", width=20,
                            height=1, font=("Calisto MT", 10, "bold"))
            lblImg1.place(x=106, y=257)

            clasificacionImg = Label(top, text="", bg="#69a3df", relief="flat",
                                     cursor="hand2", width=22, height=1, font=("Calisto MT", 10, "bold"))
            clasificacionImg.place(x=100, y=418)
            clasificacionImg.configure(text="")


            imag1_prediccion = np.array(imag1_prediccion).astype('float32')
            predKNNImg1 = knn.predict(imag1_prediccion)

            clasificacionImg.configure(text=str(Clases[predKNNImg1[0]]))
        except EXCEPTION:

            elegirImagen()

def elegirImagen2():
    global imagen2
    global pathImagen

    pathImagen = filedialog.askopenfilename(filetypes=[
        ("imagen2", ".jpg"),
        ("imagen2", ".jpeg"),
        ("imagen2", ".png")])
    if len(pathImagen) > 0:
        # leer la imagen de entrada
        imagen2 = cv2.imread(pathImagen)
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
        imagenMos2 = cv2.resize(imagen2, (100, 100))
        # visualizar la imagen de entrada en la GUI
        im = Image.fromarray(imagenMos2)
        img = ImageTk.PhotoImage(image=im)
        imagenEntrada2.configure(image=img)
        imagenEntrada2.imagen = img

        # etiqueta de imagen de entrada
        lblImg2 = Label(top, text=" Imagen Entrada", bg="#69a3df", relief="flat", cursor="hand2", width=20,
                        height=1, font=("Calisto MT", 10, "bold"))
        lblImg2.place(x=950, y=432)


def clasificarPorTecnica():
    global imagen2
    global knn
    global svm
    global cnn
    imag2_prediccion = []
    clasificacionImgTec = Label(top, text="", bg="#69a3df", relief="flat", cursor="hand2", width=22,
                                height=1, font=("Calisto MT", 10, "bold"))
    clasificacionImgTec.place(x=946, y=594)
    clasificacionImgTec.configure(text="")

    imagen2 = cv2.resize(imagen2, (64, 64))
    imagenArray = np.array(imagen2)
    pixeles = imagenArray.flatten()
    imag2_prediccion.append(pixeles)
    imag2_prediccion = np.array(imag2_prediccion).astype('float32')

    # respuesta Clasificación Imagen1

    if seleccionado.get() == 1:
        knn_prediccion = knn.predict(imag2_prediccion)
        clasificacionImgTec.configure(text=str(Clases[knn_prediccion[0]]))

    if seleccionado.get() == 2:
        svm_predImg2 = svm.predict(imag2_prediccion)
        clasificacionImgTec.configure(text=str(Clases[svm_predImg2[0]]))

    if seleccionado.get() == 3:
        imagenFinal = np.expand_dims(imagen2, axis=0)
        img_pred_cnn = imagenFinal.reshape(imagenFinal.shape[0], 64, 64, 1).astype('float32')
        img_pred_cnn = img_pred_cnn / 255
        probs = cnn.predict(img_pred_cnn)[0]
        cnn_prediccion = probs.argmax(axis=0)
        clasificacionImgTec.configure(text=str(Clases[cnn_prediccion]))


def CamaraFoto():
    global foto
    global frame
    global svm
    global Clases

    try:
        clasificacionFoto = Label(top, text="", bg="#69a3df", relief="flat",
                                      cursor="hand2", width=22, height=1, font=("Calisto MT", 10, "bold"))
        clasificacionFoto.place(x=515, y=650)

        eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ojos = eyeCascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in ojos:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ojo = eyeCascade.detectMultiScale(roi_gray)
            if len(ojo) == 0:
                print("ojo no detectado")
            else:
                for (ox, oy, ow, oh) in ojo:
                    ojos_roi = roi_color[oy:oy + oh, ox:ox + ow]

            foto = cv2.cvtColor(ojos_roi, cv2.COLOR_BGR2GRAY)
            fotoMos = cv2.resize(foto, (100, 100))
            print(foto.shape)
            im = Image.fromarray(fotoMos)
            img = ImageTk.PhotoImage(image=im)
            etiqFoto.configure(image=img)
            etiqFoto.imagen = img
                # etiqueta de imagen de entrada
            lblFoto = Label(top, text=" Imagen de Entrada", bg="#69a3df", relief="flat", cursor="hand2", width=20,
                                height=1, font=("Calisto MT", 10, "bold"))
            lblFoto.place(x=365, y=580)
            foto_gray = cv2.resize(foto, (64, 64))

            imagenFinal = np.expand_dims(foto_gray, axis=0)
            img_predCNN = imagenFinal.reshape(imagenFinal.shape[0], 64, 64, 1).astype('float32')
            img_predCNN = img_predCNN / 255
            probsFoto = cnn.predict(img_predCNN)[0]
            cnn_prediccionFoto = probsFoto.argmax(axis=0)
            clasificacionFoto.configure(text=str(Clases[cnn_prediccionFoto]))
    except:
        videoStream()



def videoStream():
    finalizar()
    global WebCam
    WebCam = cv2.VideoCapture(1)
    iniciarVideo()


def videoStream2():
    finalizar()
    global VideoRT
    VideoRT = cv2.VideoCapture(1)
    iniciarVideoRealTime()


def iniciarVideo():
    global WebCam
    global frame
    ret, frame = WebCam.read()
    if ret == True:
        etiqWebCam.place(x=365, y=116)
        frame = imutils.resize(frame, width=475)
        frameMos = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frameMos)
        image = ImageTk.PhotoImage(image=img)
        etiqWebCam.configure(image=image)
        etiqWebCam.image = image
        etiqWebCam.after(10, iniciarVideo)


def iniciarVideoRealTime():
    global VideoRT
    global imagenRT
    global Clases
    global frame
    global cnn
    global cont
    global frecuencia
    global duracion
    try:
        RT_prediccion = []
        ret, frame = VideoRT.read()
        if ret == True:
            etiqVideoRT.place(x=365, y=116)
            frame = imutils.resize(frame, width=475)
            eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ojos = eyeCascade.detectMultiScale(gray, 1.1, 4)
            for x, y, w, h in ojos:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ojo = eyeCascade.detectMultiScale(roi_gray)
                if len(ojo) == 0:
                    print("ojo no detectado")
                else:
                    for (ox, oy, ow, oh) in ojo:
                        ojos_roi = roi_color[oy:oy + oh, ox:ox + ow]



            imagenRT = cv2.cvtColor(ojos_roi, cv2.COLOR_BGR2GRAY)
            imagenRT = cv2.resize(imagenRT, (64, 64))


            imagenFinal = np.expand_dims(imagenRT, axis=0)

            RT_prediccion = imagenFinal.reshape(imagenFinal.shape[0], 64, 64, 1).astype('float32')
            RT_prediccion =  RT_prediccion/ 255
            probsRT = cnn.predict(RT_prediccion)[0]
            cnn_predRT = probsRT.argmax(axis=0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            print(cnn_predRT)
            if int(cnn_predRT) == 3:
                estado = "Ojos Abiertos"
                cv2.putText(frame,
                            estado,
                            (200, 30),
                            font, 0.5, (0, 255, 0), 1, cv2.LINE_4)
                x1, y1, w1, h1, = 0, 0, 175, 75
                # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0, 0), -1)
                cv2.putText(frame, 'escasa posibilidad de dormirse', (115, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif int(cnn_predRT) == 1:
                cont += 1
                estado = "Ojos Cerrados"
                cv2.putText(frame,
                            estado,
                            (200, 30),
                            font, 0.5, (0, 0, 255), 1, cv2.LINE_4)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0, 0), -1)
                cv2.putText(frame, 'elevada posibilidad de dormirse', (115, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if cont > 10:
                    winsound.Beep(frequency=frecuencia, duration=duracion)
                    cont = 0
            elif int(cnn_predRT) == 0:
                estado = "Ojos con Ojeras"
                cv2.putText(frame,
                            estado,
                            (200, 30),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_4)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0, 0), -1)
                cv2.putText(frame, 'Aparición de Ojeras', (115, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif int(cnn_predRT) == 2:
                estado = "Ojos Rojos"
                cv2.putText(frame,
                            estado,
                            (200, 30),
                            font, 0.5, (255, 255, 0), 1, cv2.LINE_4)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)

                # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0, 0), -1)
                cv2.putText(frame, 'Enrojecimiento de los ojos', (115, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif int(cnn_predRT) == 4:
                estado = "Ojos entre Abiertos"
                cv2.putText(frame,
                            estado,
                            (200, 30),
                            font, 0.5, (0, 0, 255), 1, cv2.LINE_4)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

                # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0, 0), -1)
                cv2.putText(frame, 'Mediana posibilidad de dormirse', (115, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            frameMos = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frameMos)
            image = ImageTk.PhotoImage(image=img)
            etiqVideoRT.configure(image=image)
            etiqVideoRT.image = image
            etiqVideoRT.after(10, iniciarVideoRealTime)
    except:
        videoStream2()


def finalizar():
    etiqWebCam.place_forget()
    etiqVideoRT.place_forget()
    if ((WebCam == None) and (VideoRT == None)):
        pass
    elif ((WebCam != None) and (VideoRT != None)):
        WebCam.release()
        VideoRT.release()
    elif (WebCam == None and VideoRT != None):
        VideoRT.release()
    elif (WebCam != None and VideoRT == None):
        WebCam.release()

def LimpiarVentana():
    imagenEntrada.place_forget()
    imagenEntrada2.place_forget()
    if ((imagenEntrada == None) and (imagenEntrada2 == None)):
        pass
    elif ((imagenEntrada != None) and (imagenEntrada2 != None)):
        imagenEntrada.release()
        imagenEntrada2.release()
    elif (imagenEntrada == None and imagenEntrada2 != None):
        imagenEntrada2.release()
    elif (imagenEntrada != None and imagenEntrada2 == None):
        imagenEntrada.release()

def ventanaAyuda():
    win = Toplevel()
    win.geometry("481x680+430+20")
    win.resizable(width=False, height=False)
    win.title("Manual de Usuario")
    fondoWin = PhotoImage(file="Manual de Usuario.png")
    fondoWin1 = Label(win, image=fondoWin).place(x=0, y=0, relwidth=1, relheight=1)

    btnCerrar2 = Button(win, text="Cerrar", bg=fondoBotones, relief="flat",
                       cursor="hand2", width=10, height=1, font=("Calisto MT", 10, "bold"), command=win.destroy)
    btnCerrar2.place(x=337, y=640)

    win.mainloop()



imagen = None
imagen2 = None
imagenRT = None
WebCam = None
VideoRT = None
foto = None
frame = None
cont = 0
frecuencia = 1500
duracion = 1500

Clases = ["Aparicion de Ojeras", "Elevada posibilidad de dormirse", "Enrojecimiento de los ojos",
          "Escasa posibilidad de dormirse", "Mediana posibilidad de dormirse"]

svm = joblib.load('Modelo_svm.joblib')
knn = joblib.load('Modelo_knn.pkl')
cnn = load_model("Modelo_cnn.h5")

imag_prediccion = []
pathImagen = ""

"""ventana principal"""
top = Tk()
top.geometry("1209x680+70+20")
top.resizable(width=False, height=False)
top.title("Sistema de detección de somnolencia")
fondo = PhotoImage(file="Sistema de deteccion.png")
fondo1 = Label(top, image=fondo).place(x=0, y=0, relwidth=1, relheight=1)



"""Columna 0"""
# etiqueta para la imagen de entrada columna 0
imagenEntrada = Label(top)
imagenEntrada.place(x=140, y=300)
# Colores
fondoBotones = "#5271ff"
fondoVideo = "#5271ff"
# Boton para imagen de entrada COLUMNA 0
btnSubirImagen = Button(top, text="Cargar Imagen", bg=fondoBotones, relief="flat",
                        cursor="hand2", width=15, height=1, font=("Calisto MT", 12, "bold"), command=elegirImagen)
btnSubirImagen.place(x=98, y=171)

btnAyuda = Button(top, text="AYUDA", bg="#3e3bcb", relief="flat",
                  cursor="hand2", width=8, height=1, font=("Calisto MT", 10, "bold"), command=ventanaAyuda)
btnAyuda.place(x=1125, y=10)

btnLimpiar = Button(top, text="Limpiar", bg=fondoBotones, relief="flat",
                  cursor="hand2", width=10, height=1, font=("Calisto MT", 10, "bold"), command=LimpiarVentana)
btnLimpiar.place(x=74, y=583)

btnCerrar = Button(top, text="Cerrar", bg=fondoBotones, relief="flat",
                  cursor="hand2", width=10, height=1, font=("Calisto MT", 10, "bold"), command=top.destroy)
btnCerrar.place(x=214, y=583)
"""Columna 1"""

# etiqueta video
etiqWebCam = Label(top)
etiqWebCam.place(x=365, y=116)

etiqVideoRT = Label(top)
etiqVideoRT.place(x=365, y=116)

etiqFoto = Label(top)
etiqFoto.place(x=557, y=533)

# Boton para iniciar la cámara
btnIniciarCam = Button(top, text="Iniciar WebCam", bg=fondoVideo, relief="flat",
                       cursor="hand2", width=12, height=1, font=("Calisto MT", 10, "bold"), command=videoStream)
btnIniciarCam.place(x=354, y=485)

# Boton para capturar foto
btnTomarFoto = Button(top, text="Tomar Foto", bg=fondoVideo, relief="flat",
                      cursor="hand2", width=11, height=1, font=("Calisto MT", 10, "bold"), command=CamaraFoto)
btnTomarFoto.place(x=484, y=485)

# Boton para iniciar video
btnVideoRT = Button(top, text="Iniciar Video RT", bg=fondoVideo, relief="flat",
                    cursor="hand2", width=11, height=1, font=("Calisto MT", 10, "bold"), command=videoStream2)
btnVideoRT.place(x=622, y=485)

# Botón para finalizar video
btnfinVideo = Button(top, text="Finalizar Video", bg=fondoVideo, relief="flat",
                     cursor="hand2", width=11, height=1, font=("Calisto MT", 10, "bold"), command=finalizar)
btnfinVideo.place(x=747, y=485)

"""columna 2"""

# Boton para imagen de entrada COLUMNA 2
btnSubirImagen2 = Button(top, text="Cargar Imagen", bg=fondoBotones, relief="flat",
                         cursor="hand2", width=15, height=1, font=("Calisto MT", 12, "bold"), command=elegirImagen2)
btnSubirImagen2.place(x=955, y=195)

# RadioButton para la elección de las tecnicas
seleccionado = IntVar()
btb_knn = Radiobutton(top, text="KNN", bg="#c7d0d8", relief="flat", cursor="hand2", width=7,
                      height=1, font=("Calisto MT", 10, "bold"), value=1, variable=seleccionado,
                      command=clasificarPorTecnica)
btb_svm = Radiobutton(top, text="SVM", bg="#c7d0d8", relief="flat", cursor="hand2", width=7,
                      height=1, font=("Calisto MT", 10, "bold"), value=2, variable=seleccionado,
                      command=clasificarPorTecnica)
btb_cnn = Radiobutton(top, text="CNN", bg="#c7d0d8", relief="flat", cursor="hand2", width=7,
                      height=1, font=("Calisto MT", 10, "bold"), value=3, variable=seleccionado,
                      command=clasificarPorTecnica)
btb_knn.place(x=978, y=273)
btb_svm.place(x=978, y=322)
btb_cnn.place(x=978, y=372)
# etiqueta para la imagen de entrada columna 2
imagenEntrada2 = Label(top)
imagenEntrada2.place(x=985, y=473)

top.mainloop()
