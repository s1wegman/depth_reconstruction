import depth_reconstruction as depth
import cv2 as cv

if __name__ == "__main__":

    leftImage = cv.imread("./datasets/crates/crates10_L.png")
    rightImage = cv.imread("./datasets/crates/crates10_R.png")

    leftImage = cv.cvtColor(leftImage, cv.COLOR_BGR2GRAY)
    rightImage = cv.cvtColor(rightImage, cv.COLOR_BGR2GRAY)

    #Specifications of IntelRealSense Camera used
    fx = 596.230590820313
    fy = 596.230590820313
    B = 0.0549554191529751
    cx = 319.500213623047
    cy = 234.513290405273

    Estimator = depth.Estimator(leftImage, rightImage)
    Estimator.setCameraParameters(fx, fy, B, cx, cy)
    Estimator.calculateDisparity(6, 6)
    Estimator.displayDisparity()

    Estimator.createDepthMap()
    Estimator.displayDepthMap()

    Estimator.createPointCloud()
    Estimator.displayPointcloud()