import os
import matplotlib.pyplot as plt 
import cv2
import numpy as np

   
if __name__ == "__main__":

    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(7.5,6))
    def update(epoch):
        name = "At_Epoch_"+str(epoch)+".png"
        print(name)
        os.chdir("records")
        plt.imshow(mpimg.imread(name))
        plt.axis('off')
        os.chdir("../")
    anim = animation.FuncAnimation(fig, update, range(0, 8000, 200), interval=175)
    name_anim = 'anim.gif'
    anim.save(name_anim)

    




    


