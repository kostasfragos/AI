import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
from model import CNN, load_model

pencil_radius = 10

MAIN_FONT = ("Comic Sans MS", 16)
MAIN_FONT_2 = ("Comic Sans MS", 8)

def on_drag(event):
    canvas.create_oval(
        event.x - pencil_radius, event.y - pencil_radius,
        event.x + pencil_radius, event.y + pencil_radius,
        fill = "white", outline = "white"
    )

def clear_canvas():
    canvas.delete("all")


ai_model = load_model()

def recognize():
    canvas.update()
    pad = 4
    x = window.winfo_rootx()+canvas.winfo_x() + pad
    y = window.winfo_rooty()+canvas.winfo_y() + pad
    x1 = x + canvas.winfo_width() - 2*pad
    y1 = y + canvas.winfo_height() - 2*pad
    # lhspsh eikonas apo canva
    im = ImageGrab.grab().crop((x,y,x1,y1))
    #ΞΞ΅ΟΞ±ΟΟΞΏΟΞ· ΟΞ΅ Ξ±ΟΟΟΞΏΞΌΞ±ΟΟΞΏ
    im = im.convert("L")
    #allagh mege8ous se 28x28 pixel
    im = im.resize((28,28))
    result, pososta = ai_model.predict_img(im)
    output_label.config(text = f'I think you draw a {result}')
    
    
    string_result = "Confidence:\n"
    for i, pososto in enumerate(pososta):
        string_result += f'{i}. {pososto*100:5.2f}%\n'

    pososta_label.config(text=string_result)
    print(string_result)

window = tk.Tk()
window.geometry("400x400")
window.title("Mnist Recognizer")
window.resizable(False, False)

pososta_label = tk.Label(window, bg='#f5f5dc', font=MAIN_FONT)
pososta_label.pack(side=tk.RIGHT, fill=tk.BOTH)

canvas = tk.Canvas(window, bg = "black")
canvas.bind("<B1-Motion>", on_drag)
canvas.pack(fill=tk.BOTH, expand = True)

clear_button = tk.Button(window, text = "Clear", bg="lightgreen", command=clear_canvas, font=MAIN_FONT_2)
clear_button.pack()

recognize_button = tk.Button(window,text = "Recognize", bg = "lightblue", command=recognize, font=MAIN_FONT_2)
recognize_button.pack()

output_label = tk.Label(window, bg='pink', font=MAIN_FONT_2)
output_label.pack()

recognize()


window.mainloop()