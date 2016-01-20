#!/usr/bin/env python
################################################################################
#                                                                              
# clicker.py - Show image and click coordinates on screen 
#
# Copyright: Carnegie Mellon University
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
#################################################################################

import Tkinter as tk 
# NOTE: class ImageTK requires python-imaging-tk (in apt-get)
import Image, ImageTk
import sys


class Clicker(object):
    coords = None

    # ----------------------------------------------------------------------- #
    def __init__(self):
        self.coords = list()

    # ----------------------------------------------------------------------- #
    def printcoords(self, event):
        """function to be called when mouse is clicked"""

        #outputting x and y coords to console
        self.coords.append((event.x, event.y))
        print (event.x,event.y)

    # ----------------------------------------------------------------------- #
    def clicker (self, filename):
        root = tk.Tk()

        coords = list()

        #setting up a tkinter canvas with scrollbars
        frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=tk.E + tk.W)
        yscroll = tk.Scrollbar(frame)
        yscroll.grid(row=0, column=1, sticky=tk.N + tk.S)
        canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)
        frame.pack(fill=tk.BOTH,expand=1)

        #adding the image
        img = ImageTk.PhotoImage(Image.open(filename))
        canvas.create_image(0,0,image=img,anchor="nw")
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        #mouseclick event
        canvas.bind("<Button 1>", self.printcoords)

        root.mainloop()



# --------------------------------------------------------------------------- #
    if __name__ == "__main__":
        
        if len(sys.argv) == 2:
            clicker = Clicker()
            clicker(sys.argv[1])
        else:
            print("Usage: clicker.py <image_filename>")
