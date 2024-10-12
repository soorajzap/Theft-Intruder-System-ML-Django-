from tkinter import*
from PIL import Image, ImageTk
from mixcode import record
from successpose2 import pose

window = Tk()
window.title("Theft Intrusion Camera")
window.iconphoto(False, PhotoImage(file='security-camera.png'))
window.geometry('1080x600')


mainFrame = Frame(window, bd=2)

label_title = Label(mainFrame, text = "Theft Intrusion Camera", font=('Helvitica', 40, 'bold'))
label_title.grid(pady=(10,10), column=2)

icon_1 = Image.open('hacker.png')
icon_1 = icon_1.resize((120, 120),Image.LANCZOS)
icon_1 = ImageTk.PhotoImage(icon_1)
label_icon_1 = Label(mainFrame, image=icon_1)
label_icon_1.grid(row=1, pady=(5, 10), column=2)

btn_image = Image.open('record.png')
btn_image = btn_image.resize((50, 50),Image.LANCZOS)
btn_image = ImageTk.PhotoImage(btn_image)
# label_btn_image = Label(mainFrame, image=btn_image)
# label_btn_image.grid(row=2, pady=(5, 10), column=2)

btn=Button(mainFrame, text="  Start Camera 1",font=('Helvitiva', 20, 'bold'), height=75, width=320, fg='green', image=btn_image, compound='left', bg='lightblue', command=record)
btn.grid(row=2, pady=(20,10), column=2)

btn=Button(mainFrame, text="  Start Camera 2",font=('Helvitiva', 20, 'bold'), height=75, width=320, fg='green', image=btn_image, compound='left', bg='lightblue', command=pose)
btn.grid(row=3, pady=(20,10), column=2)

btn_image_1 = Image.open('logout.png')
btn_image_1 = btn_image_1.resize((50, 50),Image.LANCZOS)
btn_image_1 = ImageTk.PhotoImage(btn_image_1)

btn_exit=Button(mainFrame, text=" Exit ",font=('Helvitiva', 20, 'bold'), height=75, width=320, fg='red', image=btn_image_1, compound='left', bg='lightblue', command=window.quit)
btn_exit.grid(row=4, pady=(20,10), column=2)

mainFrame.pack()



window.mainloop()
