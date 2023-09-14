# Part 1 Hough Transform

Hough transform is a one of the feature extraction techniques in computer vision to detect the lines, circles, other parametric curves in the image. Here, we have used Hough transform to detect 5 parallel lines in an image.

#### Algorithm steps:
1) Open an input image
2) Covert the image to grayscale
3) Canny edge detection- that can sketch the edges(here lines) in the image by suppressing noise
4) Hough Transform- This function calculates H, rhos, thetas and detect the lines in the image
5) Getting the indices of the hough peaks- It is used to identify the peaks(local maxima) in the Hough accumulator
6) Hough lines draw- This function is used to draw hough lines from the indices obtained

#### Experiments performed:
1) We experimented with different values of epsilon values to determine the neighborhood to search for the thetas. As the image is not perfect we cannot expect all the five lines to be perfectly parallel. As a result we need to search within a neighborhood of the theta value.
2)While getting the  parallel lines it is not necessary that all the five lines corrospond to global maxima in the Hough space. We can have lines of different orientation corrosponding to the global maxima. Additionally if we detect a global maxima the next point having the most votes could be near the maxima. To combat this we need to set the neighbouring values to 0. The parameter of neigborhood_size was experimented with to decide how much of the neighborhood to mute.
3)Experimented on line thickness to see the changes happening while detecting lines.
4) We also tried the open cv hough transform library to see what lines they are detecting in the image. Surprisingly they detected horizontal lines in the image.
5) We also experimented with the different thresholds for the canny and sobel filtering.

#### Difficulties faced:
1)The main difficulty that we faced was there were many horizontal lines in the image and edges for the image were getting detected and so it was very difficult to suppress them and thats why we had to keep the threshold for canny edge very high becuase for low threshold horizontal lines were getting detected.

2)All the five parallel lines were not necessarily global maximum. Some of them were global maximum and some of them were local maximum. Identifying the local maximum was a challange. To combat this challange we had to use a high value of canny threshold so that all the background lines were muted. In addition we had to suppress the values across each global maximum so that other local maximum could be detected. 

### Observations

As it can be seen in the hough space the sinsusoidal waves intersect. We can see set of five intersections which corresponds to the five lines to be detected. I have commented the code which views the hough space though I have attached a screenshot of the hough space for the sample-input.png image. To view hough space for other images uncomment the code.


#### What we can do further to make our algorithm better :
1) We tried to overcome the above difficulty by finding the global maxima for the first line and then using the same orientation, we found the orientation for the other lines.
2) Checking multiple peaks allow us to focus on different orientation of line.

#### Output:

Final Output: Staff of lines detected

<img width="661" alt="Screen Shot 2023-02-21 at 2 44 06 PM" src="https://media.github.iu.edu/user/21193/files/129d7975-8a75-4b87-a832-5d8a3a971ab4">


Hough Plot:

<img width="255" alt="Screen Shot 2023-02-21 at 2 43 51 PM" src="https://media.github.iu.edu/user/21193/files/ce90b931-97b8-4a53-9e44-48c9eb705b3c">


Hough Space:

<img width="200" alt="Screen Shot 2023-02-21 at 2 56 29 PM" src="https://media.github.iu.edu/user/21193/files/129ccd99-ee3a-414c-9d02-053128ad48fe">

