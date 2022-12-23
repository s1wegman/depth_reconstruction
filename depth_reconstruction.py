import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import MCP, MCP_Flexible
import open3d as o3d
from joblib import Parallel, delayed
import scipy
from scipy.interpolate import UnivariateSpline

"""This class adds an occlusion penalty by overriding parts of the pathfinding
algorithm. It can be omitted, if this occlusion penalty is not desired."""
class FlexibleMCP(MCP_Flexible):
    
        def travel_cost(self, old_cost, new_cost, offset_length):
           if offset_length > 1:
               return new_cost + 0.025
           else:
               return new_cost

class Estimator():
    def __init__(self, imageL, imageR):
        self.imageL = imageL
        self.imageR = imageR
        
    def setCameraParameters(self, focalLengthX, focalLengthY, baseline, centerX, centerY):
        self.focalLengthX = focalLengthX
        self.focalLengthY = focalLengthY
        self.baseline = baseline
        self.centerX = centerX
        self.centerY = centerY

    def calculateDisparity(self, templateWidth, templateHeight, dsiRange=256, varThreshold=5, seedThreshold=0.025, forceSeedpoints=True, splineRange=3):
        self.splineRange = splineRange
        self.dsiRange = dsiRange
        self.forceSeedpoints = forceSeedpoints
        self.varThreshold = varThreshold
        self.seedThreshold = seedThreshold
        
        disparity = np.zeros_like(self.imageL, dtype=np.float32)
        
        self.imageL = np.pad(self.imageL, [(int(templateHeight/2),), (int(templateWidth/2),)], mode='constant')
        self.imageR = np.pad(self.imageR, [(int(templateHeight/2),), (int(templateWidth/2),)], mode='constant')
        
        def f(row):
            lineDisp = self.calculateLineDisparity(templateWidth, templateHeight, row)
            return lineDisp
        
        # Consecutive execution
        # for x in range(0, self.imageL.shape[0]-tHeight):
        #     disparity[x,:] = f(x)
        
        #Parallel execution
        lineDisparities = Parallel(n_jobs=-1)(delayed(f)(row) for row in range(0, self.imageL.shape[0]-templateHeight))
        for idx, line in enumerate(lineDisparities):
            disparity[idx,:] = line
        
        self.disparity = disparity
    
    def calculateLineDisparity(self, tWidth, tHeight, ix):
        lineDSI = np.ones((self.dsiRange, self.imageL.shape[1]-tWidth), dtype=np.float32)
        lineVariance = []
        offsetDSI = 0
        for iy in range(0, self.imageL.shape[1]-tWidth):
            target = self.imageR[ix:ix+tHeight, :iy+tWidth]
            template = self.imageL[ix:ix+tHeight, iy:iy+tWidth]
            res = (cv.matchTemplate(target, template, cv.TM_SQDIFF_NORMED)).flatten()
            
            # Variance of every template for texture threshold
            lineVariance.append(np.var(template))
            
            if iy >= self.dsiRange:
                lineDSI[:,iy] = res[offsetDSI:iy][::-1]
                offsetDSI += 1
            else:
                lineDSI[:len(res),iy] = res[::-1]
        
        traceback = self.findCostPath(lineDSI, lineVariance)
        lineDisparity, lOccl, rOccl = self.findDisparityFromPath(lineDSI, traceback)
        lineDisparity = self.fillOcclusions(lineDisparity, lOccl, rOccl)

        return lineDisparity

    def findCostPath(self, cost, lineVariance):
        r = cost.shape[0]
        c = cost.shape[1]
        
        # Setting the cost of seed points to zero
        mask = self.findCandidates(cost, lineVariance)
        cost = np.where(mask, 0, cost)
        
        # Forcing the path through seed points by setting all other entries to 1
        if self.forceSeedpoints:
            for column in range(c):
                if (np.any((cost[:,column] == 0))):
                    cost[:,column] = [1 if i != 0 else 0 for i in cost[:,column]]
        
        # Directions, the pathfinding algorithm can take
        offsets=[(-1,1), (1,1), (0,1)]
        M = FlexibleMCP(cost,offsets=offsets)
        
        # Minimum cost path for every entry in last column
        ends = [[i,c-1] for i in range(r)]
        cost_array, traceback_array = M.find_costs([[0,0]], ends)
        ends_idx = tuple(np.asarray(ends).T.tolist())
        min_cost = np.argmin(cost_array[ends_idx])
        traceback = M.traceback([min_cost,c-1])

        # Minimum cost path to top right corner
        # cost_array, traceback_array = M.find_costs([[0,0]], [[0,c-1]])
        # traceback = M.traceback([0,c-1])

        return traceback

    def findDisparityFromPath(self, cost, traceback):
        c = cost.shape[1]

        disp = np.zeros((c,), dtype=np.float32)
        rOccl = []
        lOccl = []

        prevX = 0

        for x in range(len(traceback)):

            # Left Occlusion (only in left)
            if (traceback[x][0] > prevX):
                lOccl.append(traceback[x][1])

            # Right Occlusion (only in right)
            elif (traceback[x][0] < prevX):
                rOccl.append(traceback[x][1])

            # Matched Pixel
            else:
                # Get cost entries around every path entry and fit spline
                costLine = cost[traceback[x][0]-self.splineRange:traceback[x][0]+self.splineRange+1, traceback[x][1]]
                disparityOffset = self.getSubpixelDisparity(costLine)
                disp[traceback[x][1]] = np.abs(traceback[x][0] + disparityOffset)

            prevX = traceback[x][0]

        return disp, lOccl, rOccl

    def fillOcclusions(self, lineDisparity, lOccl, rOccl):
        
        for oclL in lOccl:
            lineDisparity[oclL] = lineDisparity[oclL-1]
            
        for oclR in reversed(rOccl):
            try:
                lineDisparity[oclR] = lineDisparity[oclR+1]
            except IndexError:
                lineDisparity[oclR] = 0
        
        return lineDisparity
    
    def findCandidates(self, cost, lineVariance):
        rows = cost.shape[0]
        columns = cost.shape[1]
        mask = np.full((rows, columns), False)
        for c in range(columns):
            seedC = np.nanargmin(cost[:,c])
            seedD = np.argmin(np.diag(cost, k=c-seedC))
            if (cost[seedC, c] <= self.seedThreshold):
                if cost[seedC, c] == cost[seedD,c]:
                    if lineVariance[c] > self.varThreshold:
                        mask[seedC, c ] = True
                        
        mask = self.removeLoneCandidates(mask)
        return mask
                
    def removeLoneCandidates(self, mask):
        W = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        convolution = scipy.signal.convolve2d(mask, W,'same')
        mask = np.logical_and(convolution >= 2, mask)
        
        return mask
        
    def getSubpixelDisparity(self, costs):
        # Check if there are zeros or not enough entries for spline fitting
        if not any(costs) or self.splineRange*2 > len(costs):
            disparityOffset = 0
            return disparityOffset

        else:
            x = np.arange(len(costs))
            y = costs
            try:
                spl = UnivariateSpline(x, y, k=4, s=0)
                roots = spl.derivative().roots()
                globalMinimum = roots[np.argmin(spl(roots))]
            except ValueError:
                disparityOffset = 0
                return disparityOffset

            disparityOffset = globalMinimum - self.splineRange
            return disparityOffset

    def createPointCloud(self):
        depthMap = self.depthMap
        fx = self.focalLengthX
        fy = self.focalLengthY
        B = self.baseline
        cx = self.centerX
        cy = self.centerY
        h = depthMap.shape[0]
        w = depthMap.shape[1]
        
        depth_image = o3d.geometry.Image(depthMap.astype(np.uint16))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,cx,cy)
        intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = np.array([[1,0,0, B],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                              cam.intrinsic, 
                                                              cam.extrinsic,
                                                              1000.0,
                                                              5.0
                                                              )

        # For point clouds with color data
        # imagePath = " "
        # color_raw = o3d.io.read_image(imagePath)
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_image, 1000, 2)
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        #                                                       cam.intrinsic, 
        #                                                       cam.extrinsic,
        #                                                       )

        self.pointCloud = pcd
        
    def displayPointcloud(self):
        o3d.visualization.draw_geometries([self.pointCloud])
        
    def createDepthMap(self):
        disparity = self.disparity
        fx = self.focalLengthX
        B = self.baseline

        depth = np.zeros_like(disparity, dtype=np.uint16)
        h = disparity.shape[0]
        w = disparity.shape[1]

        for row in range(h):
            for column in range(w):
                Z = fx*B/disparity[row, column]
                depth[row,column] = Z*1000

        self.depthMap = depth
        
    def displayDisparity(self):
        plt.figure()
        plt.imshow(self.disparity)
    
    def displayDepthMap(self):
        #Factor to display units as meters
        plt.figure()
        plt.imshow(self.depthMap/1000)
        plt.colorbar(location='bottom')

    # Intital test done for disparity voting.

    # def voteDisparity(self, image):
    #     disparity = self.disparity
    #     for row in range(image.shape[0]):
    #         segments = self.createSegments(image[row,:], 10, 640)
    #         lineVote = self.voteDisparityLine(disparity[row,:], segments)
    #         disparity[row,:] = lineVote

    #     self.disparity = disparity

    # def voteDisparityLine(self, disparityRow, segments):
    #     for x in range(len(segments)):
    #         votes = {}
    #         if (disparityRow.all == 0):
    #                 winner = 0

    #         else:
    #             for i in range(segments[x][0],segments[x][1]+1):
    #                 votes.setdefault(disparityRow[i], 0)
    #                 if (disparityRow[i] != 0):
    #                     votes[disparityRow[i]] += 1
    #             try:
    #                 winner = max(votes, key=votes.get)
    #                 disparityRow[segments[x][0]:segments[x][1]] = winner
    #             except ValueError:
    #                 continue

        # return disparityRow

    # def createSegments(self, imageRow, colorDifference, maxLength):
        
    #     startY = 0
    #     L = 0
    #     segment = []
    
    #     for y in range(len(imageRow)-1):
    #         L+=1
    #         if np.abs(int(imageRow[y]) - int(imageRow[y+1])) > colorDifference or L >= maxLength:
    #             segment.append([startY, y])
    #             startY = y+1
    #             L = 0

    #     segment.append([startY, y])
    #     return segment

    # Initial testing to fit splines to segments with similar disparity

    # def fitDisparitySegments(self, disp):
    #     segments = self.createSegments(disp, 0.25, 200)
    #     new_disp = np.zeros_like(disp)
    #     for i in range(len(segments)):
    #         x = np.arange(segments[i][1]-segments[i][0])
    #         y = disp[segments[i][0]:segments[i][1]]
    #         if not any(y == 0) and len(y) > 3:
    #             spl = UnivariateSpline(x, y, k=1, s=len(x))
    #             new_disp[segments[i][0]:segments[i][1]] = spl.__call__(x)
    #     return new_disp