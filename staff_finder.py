## References used -> https://www.geeksforgeeks.org/python-edge-detection-using-pillow/ , https://www.geeksforgeeks.org/python-pil-image-new-method/
## https://www.geeksforgeeks.org/python-pil-image-new-method/, https://www.geeksforgeeks.org/how-to-manipulate-the-pixel-values-of-an-image-using-python/
## https://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/, https://www.geeksforgeeks.org/python-pil-imagedraw-draw-line/
## https://www.geeksforgeeks.org/python-pil-imagedraw-draw-line/, https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
## https://numpy.org/doc/stable/reference/generated/numpy.argmax.html, https://www.geeksforgeeks.org/how-to-compare-two-numpy-arrays/
## https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92, https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
## https://www.geeksforgeeks.org/python-pil-imagedraw-draw-line/, https://forum.image.sc/t/how-to-draw-a-line-captured-by-the-hough-transform/32519
## https://github.com/taochenshh/Find-Lines-and-Circles-in-an-Image-Hough-Transform-/blob/master/hough_lines_draw.m, https://content.byui.edu/file/b8b83119-9acc-4a7b-bc84-efacf9043998/1/Math-2-11-2.html
## https://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/, https://www.geeksforgeeks.org/python-edge-detection-using-pillow/
## https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123, https://www.analyticsvidhya.com/blog/2022/06/a-complete-guide-on-hough-transform/

import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw
import sys
import matplotlib.pyplot as plt

def test_function(img):
    width, height = img.size
    image_new = np.zeros(shape=(height,width))

    for i in range(height):
        for j in range(width):
            x = img.getpixel((j,i))
            image_new[i,j] = x

    image_new = Image.fromarray(image_new)     


## Reference for this is the slides
## Sobel operation on the image
def sobel(img):
    image_x = img.filter(ImageFilter.Kernel((3, 3), (-1/8, 0, 1/8, -2/8, 0, 2/8, -1/8, 0, 1/8), 1, 0))
    #image_x.show()

    image_y = img.filter(ImageFilter.Kernel((3, 3), (1/8, 2/8, 1/8, 0, 0, 0, -1/8, -2/8, -1/8), 1, 0))
    #image_y.show()

    width,height = img.size

    image_sobel = np.zeros(shape=(height,width))

    for i in range(height):
        for j in range(width):
           
            x = image_x.getpixel((j,i))
            y = image_y.getpixel((j,i))
            image_sobel[i,j] = int(((x*x) + (y*y))**0.5)

    image_sobel = Image.fromarray(image_sobel)     
    #image_sobel.show()                    
    return image_sobel

## Reference for this is the slides.
## Canny operation on the image
def canny(img,low_threshold=50,high_threshold=150):
    x_dir = [-1,0,1,1,1,0,-1,-1]
    y_dir = [1,1,1,0,-1,-1,-1,0]
    image_sobel = sobel(img)

    width, height = image_sobel.size

    image_canny = np.zeros(shape=(height,width))
    for i in range(height):
        for j in range(width):
            x = image_sobel.getpixel((j,i))

            if(x>=high_threshold):
                image_canny[i,j] = 255
            elif(x>=low_threshold):
                image_canny[i,j] = 0
                for k in range(8):
                    if(i+k>=0 and i+k<height and j+k>=0 and j+k<width and image_sobel.getpixel((j+k,i+k))>0):
                        image_canny[i,j] = 255
            else:
               image_canny[i,j] = 0 
            
            if(i==0 or i==height-1 or j==0 or j==width-1):
                image_canny[i,j] = 0 
    
    image_canny = Image.fromarray(image_canny) 
    #image_canny.show()
    return image_canny

def laplacian(img):
    image_laplacian = img.filter(ImageFilter.FIND_EDGES)
    return image_laplacian


## Reference for this code -> https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
## https://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/
## https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92 
## Concept on how to code hough transform has been understood from the above link.
## This function makes the hough accumulator
def hough(image):
    height, width = image.shape 
    
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) 
    
    ## we are taking our rho to be from -image_diagonal to +image_diagonal as that is the maximum length we can see in out image space
    rho = np.arange(-img_diagonal, img_diagonal + 1, 1)
    theta = np.deg2rad(np.arange(0, 361, 1))

    # create the empty Hough Accumulator with dimensions equal to the size of rhos and thetas

    H = np.zeros((len(rho), len(theta)), dtype=np.uint64)
    height_idxs, width_idxs = np.nonzero(image)
    
    ## going through non zero points
    for i in range(len(width_idxs)):
        x = width_idxs[i]
        y = height_idxs[i]

        for j in range(len(theta)): # cycle through thetas and calc rho
            one_rho = (x * np.cos(theta[j]) + y * np.sin(theta[j]))
            H[(int)(one_rho+img_diagonal), j] += 1

    hough_space = Image.fromarray((H * 255).astype(np.uint8))

    ## Uncomment the below code to view the Hough space

    # fig = plt.figure(figsize=(20, 20))
    # ## Reference -> https://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab
    # fig.canvas.manager.set_window_title('Hough Plot')
    
    # ## https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # plt.imshow(H)

    # plt.xlabel('Theta Direction')
    # plt.ylabel('Rho Direction')

    # plt.show() 

    return H, rho, theta

## References -> https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
## https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
## This fuctions detects the parallel lines from the hough accumulator.
def find_hough_peaks(H, num_peak, thetas, adjacent_pixels_size=5):
    indices = []

    theta_consider = np.deg2rad(-90)
    epsilon  = np.deg2rad(5)
    i=0

    while(i<num_peak):
        #print(i)
        idx = np.unravel_index(np.argmax(H), H.shape)

        if(i==0):
            indices.append(idx)
            i = i+1
            theta_consider = thetas[idx[1]]
        else:
            if(thetas[idx[1]]>=theta_consider-epsilon and thetas[idx[1]]<=theta_consider+epsilon):
                indices.append(idx)
                i = i+1
    
        index_y, index_x = idx

        for x in range((int)(index_x-(adjacent_pixels_size)*0.5),(int)(index_x+(adjacent_pixels_size)*0.5)):
            for y in range((int)(index_y-(adjacent_pixels_size)*0.5),(int)(index_y+(adjacent_pixels_size)*0.5)):
                if(x>=0 and x<H.shape[1] and y>=0 and y<H.shape[0]):
                    H[y,x] = 0
        

    return indices


# drawing the lines from the Hough Accumulatorlines using OpevCV document
## Reference for this part -> https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
## https://www.geeksforgeeks.org/python-pil-image-new-method/
## This fuctions draws the parallel lines on the image from the hough accumulator.
def hough_lines_draw(img, indicies, rhos, thetas):
    width,height = img.size
    img_diag = np.ceil(np.sqrt(height**2 + width**2))

    rgbimg = Image.new("RGBA", img.size)
    rgbimg.paste(img)
    img1 = ImageDraw.Draw(rgbimg)

    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]


        a = np.cos(theta)
        b = np.sin(theta)
        x_not = a*rho
        y_not = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int((x_not + 1500*(-b)))
        y1 = int((y_not + 1500*(a)))
        x2 = int((x_not - 1500*(-b)))
        y2 = int((y_not - 1500*(a)))

        img1.line([(x1, y1), (x2, y2)], fill = "red", width = 3)
        #rgbimg.show()
    return rgbimg 


if __name__ == '__main__':
    if(len(sys.argv) < 2):
        raise Exception(" pyhton command is : python3 staff_finder.py input.jpg/input.png")

    image = Image.open(sys.argv[1])

    ## Converting to grayscale
    image_grayscale = image.convert("L")

    if(sys.argv[1] == "sample-input.png"):
        image_canny = np.asarray(canny(image_grayscale, 100, 150))
    else:
        image_canny = np.asarray(canny(image_grayscale, 50, 100))

    H,rhos,thetas = hough(image_canny)
    indices = find_hough_peaks(H,5,thetas)
    
    detected_staff = hough_lines_draw(image,indices,rhos,thetas)
    #detected_lines.show()

    detected_staff.save('detected_staff.png')
