from utils import *
import cv2 as cv
import numpy as np

# # # # #                              SURF                              # # # # #

# # #       1 - Key-point detection using SURF in all four images       # # #

surf = cv.xfeatures2d_SURF.create(3500)

# # Image 1 # #

img1 = cv.imread('NISwGSP/02.jpg')  # All images in NISwGSP have resolution: (3264, 2448)

cv.namedWindow('Cathedral_1', cv.WINDOW_NORMAL)
cv.imshow('Cathedral_1', img1)
cv.waitKey(0)

# Detect key-points in img1
kp1 = surf.detect(img1)
# Compute descriptors of key-points in img1
desc1 = surf.compute(img1, kp1)

# Draw the key-points on the image
k_img1 = cv.drawKeypoints(img1, kp1, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Cathedral_1', k_img1)
cv.waitKey(0)

# # Image 2 # #

img2 = cv.imread('NISwGSP/01.jpg')

cv.namedWindow('Cathedral_2', cv.WINDOW_NORMAL)
cv.imshow('Cathedral_2', img2)
cv.waitKey(0)

# Detect key-points in img2
kp2 = surf.detect(img2)
# Compute descriptors of key-points in img2
desc2 = surf.compute(img2, kp2)

# Draw the key-points on the image
k_img2 = cv.drawKeypoints(img2, kp2, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Cathedral_2', k_img2)
cv.waitKey(0)

# # Image 3 # #

surf = cv.xfeatures2d_SURF.create(4000)

img3 = cv.imread('NISwGSP/03.jpg')

cv.namedWindow('Cathedral_3', cv.WINDOW_NORMAL)
cv.imshow('Cathedral_3', img3)
cv.waitKey(0)

# Detect key-points in img3
kp3 = surf.detect(img3)
# Compute descriptors of key-points in img3
desc3 = surf.compute(img3, kp3)

# Draw the key-points on the image
k_img3 = cv.drawKeypoints(img3, kp3, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Cathedral_3', k_img3)
cv.waitKey(0)

# # Image 4 # #

img4 = cv.imread('NISwGSP/04.jpg')

cv.namedWindow('Cathedral_4', cv.WINDOW_NORMAL)
cv.imshow('Cathedral_4', img4)
cv.waitKey(0)

# Detect key-points in img4
kp4 = surf.detect(img4)
# Compute descriptors of key-points in img4
desc4 = surf.compute(img4, kp4)

# Draw the key-points on the image
k_img4 = cv.drawKeypoints(img4, kp4, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Cathedral_4', k_img4)
cv.waitKey(0)

# # #     2 - Key-point matching using the "cross-checking" method and combining images     # # #

# # Match the key-points of img1 and img2 # #

matches_1_2 = match_2(desc1[1], desc2[1])

img_1_2 = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches_1_2, None)

cv.namedWindow('Matched-1-2', cv.WINDOW_NORMAL)
cv.imshow('Matched-1-2', img_1_2)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []
for x in matches_1_2:
    img_pt1.append(kp1[x.queryIdx].pt)
    img_pt2.append(kp2[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

# Calculates the Homography to connect the images appropriately
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

img_1_2 = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]+2100))
img_1_2[0: img1.shape[0], 0: img1.shape[1]] = img1

cv.imshow('Matched-1-2', img_1_2)
cv.waitKey(0)

# # Match the key-points of img3 and img4 # #

matches_3_4 = match_2(desc3[1], desc4[1])

img_3_4 = cv.drawMatches(img3, desc3[0], img4, desc4[0], matches_3_4, None)

cv.namedWindow('Matched-3-4', cv.WINDOW_NORMAL)
cv.imshow('Matched-3-4', img_3_4)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []
for x in matches_3_4:
    img_pt1.append(kp3[x.queryIdx].pt)
    img_pt2.append(kp4[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

# Calculates the Homography to connect the images appropriately
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

img_3_4 = cv.warpPerspective(img4, M, (img3.shape[1]+1000, img3.shape[0]+2300))
img_3_4[0: img3.shape[0], 0: img3.shape[1]] = img3

cv.imshow('Matched-3-4', img_3_4)
cv.waitKey(0)

cv.destroyAllWindows()

cv.namedWindow('Matched-1-2', cv.WINDOW_NORMAL)
cv.imshow('Matched-1-2', img_1_2)

cv.namedWindow('Matched-3-4', cv.WINDOW_NORMAL)
cv.imshow('Matched-3-4', img_3_4)
cv.waitKey(0)

# # Match the key-points of img_1_2 and img_3_4 # #

surf = cv.xfeatures2d_SURF.create(4250)

# Detect key-points in img_1_2
kp_1_2 = surf.detect(img_1_2)
# Compute descriptors of key-points in img1
desc_1_2 = surf.compute(img_1_2, kp_1_2)

# Draw the key-points on the image
k_img_1_2 = cv.drawKeypoints(img_1_2, kp_1_2, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Matched-1-2', k_img_1_2)
cv.waitKey(0)

# Detect key-points in img_1_2
kp_3_4 = surf.detect(img_3_4)
# Compute descriptors of key-points in img1
desc_3_4 = surf.compute(img_3_4, kp_3_4)

# Draw the key-points on the image
k_img_3_4 = cv.drawKeypoints(img_3_4, kp_3_4, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('Matched-3-4', k_img_3_4)
cv.waitKey(0)

matches_all = match_2(desc_1_2[1], desc_3_4[1])

img_all = cv.drawMatches(img_1_2, desc_1_2[0], img_3_4, desc_3_4[0], matches_all, None)

cv.namedWindow('All-Matched', cv.WINDOW_NORMAL)
cv.imshow('All-Matched', img_all)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []
for x in matches_all:
    img_pt1.append(kp_1_2[x.queryIdx].pt)
    img_pt2.append(kp_3_4[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

# Calculates the Homography to connect the images appropriately
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

img_all = cv.warpPerspective(img_3_4, M, (img_1_2.shape[1]+1800, img_1_2.shape[0]))
img_all[0: img_1_2.shape[0], 0: img_1_2.shape[1]] = img_1_2
img_all = img_all[0:img_all.shape[0]-600, :]

cv.imshow('All-Matched', img_all)
cv.waitKey(0)

cv.imwrite('NISwGSP/Stitched_Surf.jpg', img_all)

cv.destroyAllWindows()

# # #                   3 - Metrics Calculation                   # # #

# #   Calculate f2-differential entropy   # #

print("=========== Metrics ===========")

# Calculate the global entropy values for stitched and constituent images

print("")
print("1) Global entropies")
print("")

g_ent_1 = global_entropy(img1)
print("Global entropy of image 1: g_ent_1 = %.3f" % g_ent_1)

g_ent_2 = global_entropy(img2)
print("Global entropy of image 2: g_ent_2 = %.3f" % g_ent_2)

g_ent_3 = global_entropy(img3)
print("Global entropy of image 3: g_ent_3 = %.3f" % g_ent_3)

g_ent_4 = global_entropy(img4)
print("Global entropy of image 4: g_ent_4 = %.3f" % g_ent_4)

g_ent_all = global_entropy(img_all)
print("Global entropy of Panorama: g_ent_all = %.3f" % g_ent_all)
print("")

print("")
print("2) f2-differential entropy")
print("")

# Calculate f2-differential entropy
g_mean_ent = np.mean([g_ent_1, g_ent_2, g_ent_3, g_ent_4])
f2 = g_ent_all - g_mean_ent

print("F2 = %.3f =====> F2 Normalized = %.3f" % (f2, f2/8))  # Max F2 = 8 = log2(2^8)
print("")

# #   Calculate f3-average local entropy for the stitched image   # #

print("")
print("3) f3-average local entropy for the stitched image")
print("")

# Calculate the local entropy array for stitched and constituent images

print("Please wait: Calculating local entropy of image 1...")
l_ent_1 = local_entropy(img1)
print("Calculation complete!")
print("")

print("Please wait: Calculating local entropy of image 2...")
l_ent_2 = local_entropy(img2)
print("Calculation complete!")
print("")

print("Please wait: Calculating local entropy of image 3...")
l_ent_3 = local_entropy(img3)
print("Calculation complete!")
print("")

print("Please wait: Calculating local entropy of image 4...")
l_ent_4 = local_entropy(img4)
print("Calculation complete!")
print("")

print("Please wait: Calculating local entropy of stitched image...")
l_ent_all = local_entropy(img_all)
print("Calculation complete!")
print("")

f3 = np.mean(l_ent_all)
print("F3 = %.3f =====> F3 Normalized = %.3f" % (f3, f3/8))  # Max F3 = 8 = log2(2^8)
print("")

# #   Calculate f4-differential variance of the local entropy   # #

print("")
print("4) f4-differential variance of the local entropy")
print("")

f4 = np.var(l_ent_1 + l_ent_2 + l_ent_3 + l_ent_4) - np.var(l_ent_all)
print("F4 = %.3f =====> F4 Normalized = %.3f" % (f4, f4/64))  # Max F4 = 64 = ( log2(2^8) )^2
print("")

# #   Calculate f9-absolute difference of standard deviations   # #

print("")
print("5) f9-absolute difference of standard deviations")
print("")

f9 = abs(np.std(img_all) - np.mean(np.std(img1) + np.std(img2) + np.std(img3) + np.std(img4)))
print("F9 = %.3f =====> F9 Normalized = %.3f" % (f9, f9/255))  # Max F9 = 255 = 2^8
print("")
