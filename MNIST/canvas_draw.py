from tkinter import *
from PIL import ImageGrab
import numpy as np
import cv2
import matplotlib.pyplot as plt
from reconstruction import process_image, get_reconstructed_image
import cv2
import datetime
b1 = "up"
xold, yold = None, None
image = np.zeros([28, 28])
reconstructed_image = np.zeros([28, 28])
predicted_number = None

def main():
    global image
    global reconstructed_image
    global predicted_number
    root = Tk()
    root.title("Canvas Draw")
    drawing_area = Canvas(root,width=600,height=600)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    button4=Button(root,fg="green",text="identify digit",command=lambda:getter(drawing_area))
    button4.pack(side=RIGHT)
    button4=Button(root,fg="green",text="Clear",command=lambda:delete(drawing_area))
    button4.pack(side=LEFT)


    def update_reconstructed_image(widget):
        values = np.array([slider.get() for slider in sliders], dtype="float32")
        reconstructed_image = get_reconstructed_image(values, predicted_number)
        axs[1].imshow(reconstructed_image)

    root2 = Tk()
    sliders = []
    for i in range(16):
        w = Scale(root2, from_=-1, to=1, orient=HORIZONTAL, resolution = 0.001)
        w.bind("<ButtonRelease-1>", update_reconstructed_image)
        sliders.append(w)
        w.pack()

    def delete(widget):
        widget.delete("all")


    def getter(widget):
        global image
        global reconstructed_image
        global predicted_number

        x=root.winfo_rootx()+widget.winfo_x()
        y=root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        image = ImageGrab.grab().crop((x,y,x1,y1))
        image = np.array(image)
        image = 255 - image
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255
        image = image.astype("float32")
        image = (image > .1) * image
        predicted_number, predicted_number_chance, second_number, \
        second_number_chance, predicted_number_vector, reconstructed_image = process_image(image)

        predicted_number_vector = np.round(predicted_number_vector, decimals=3)

        for i, slider in enumerate(sliders):
            slider.set(predicted_number_vector[i])

        axs[0].imshow(image)
        axs[1].imshow(reconstructed_image)

        axs[0].set_xlabel('predicted numeber: ' + str(predicted_number) + ' chance: ' + str(predicted_number_chance))
        axs[0].set_title('original number')
        axs[1].set_title('reconstructed number')

        plt.draw()

    plt.ion()
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Digits')
    axs[0].imshow(image)
    axs[1].imshow(reconstructed_image)
    plt.draw()

    root.mainloop()


def b1down(event):
    global b1
    b1 = "down"


def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None
    yold = None

def motion(event):
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_rectangle(xold,yold,event.x,event.y, width=30)
        xold = event.x
        yold = event.y


if __name__ == "__main__":
    main()
