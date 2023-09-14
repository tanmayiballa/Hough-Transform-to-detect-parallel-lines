# Hough-Transform-to-detect-parallel-lines

Hough transform is one of the feature extraction techniques in computer vision to detect the lines, circles, and other parametric curves in the image. Here, we have used the Hough transform to detect 5 parallel lines in an image.

#### Algorithm steps:
1) Open an input image
2) Covert the image to grayscale
3) Canny edge detection- that can sketch the edges(here lines) in the image by suppressing noise
4) Hough Transform- This function calculates H, rhos, and thetas and detects the lines in the image
5) Getting the indices of the Hough peaks- It is used to identify the peaks(local maxima) in the Hough accumulator
6) Hough lines draw- This function is used to draw hough lines from the indices obtained

#### Experiments performed:
1) We experimented with different values of epsilon values to determine the neighborhood to search for the thetas. As the image is not perfect we cannot expect all the five lines to be perfectly parallel. As a result, we need to search within a neighborhood of the theta value.
2)While getting the  parallel lines it is not necessary that all the five lines correspond to global maxima in the Hough space. We can have lines of different orientations corresponding to the global maxima. Additionally, if we detect a global maxima the next point having the most votes could be near the maxima. To combat this we need to set the neighboring values to 0. The parameter of neigborhood_size was experimented with to decide how much of the neighborhood to mute.
3)Experimented with line thickness to see the changes happening while detecting lines.
4) We also tried the opencv Hough transform library to see what lines they detect in the image. Surprisingly they detected horizontal lines in the image.
5) We also experimented with the different thresholds for the canny and Sobel filtering.

### Observations

As it can be seen in the hough space the sinusoidal waves intersect. We can observe a set of five intersections that correspond to the five lines to be detected.


#### What we can do further to make our algorithm better :
1) We tried to overcome the above difficulty by finding the global maxima for the first line and then using the same orientation, we found the orientation for the other lines.
2) Checking multiple peaks allow us to focus on the different orientation of the line.

#### Output:

Final Output: Staff of lines detected

![image](https://github.com/tanmayiballa/Hough-Transform-to-detect-parallel-lines/edit/main/detected.png)

| Hough Plot  | Hough Space |
| -------  | - |
| ![Image](https://github.com/tanmayiballa/Hough-Transform-to-detect-parallel-lines/edit/main/houghplot) | ![Image](https://github.com/tanmayiballa/Hough-Transform-to-detect-parallel-lines/edit/main/houghspace) |
