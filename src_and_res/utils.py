import numpy as np
import cv2 as cv
from scipy.stats import entropy


# Good matching without cross-checking
def match_2(d1, d2):
    n1 = d1.shape[0]

    matches = []
    for i in range(n1):
        line_v = d1[i, :]
        diff = d2 - line_v
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        ind_min1 = np.argmin(distances)
        min_dist1 = distances[ind_min1]

        distances[ind_min1] = np.inf

        ind_min2 = np.argmin(distances)
        min_dist2 = distances[ind_min2]

        if min_dist1 / min_dist2 < 0.5:
            matches.append(cv.DMatch(i, ind_min1, min_dist1))

    return matches


# Good matching with cross-checking
def match_1(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    possible_matches_1 = []
    for i in range(n1):
        line_v = d1[i, :]
        diff = d2 - line_v
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        ind_min1 = np.argmin(distances)
        min_dist1 = distances[ind_min1]

        distances[ind_min1] = np.inf

        ind_min2 = np.argmin(distances)
        min_dist2 = distances[ind_min2]

        if min_dist1 / min_dist2 < 0.5:
            possible_matches_1.append(cv.DMatch(i, ind_min1, min_dist1))

    matches = []
    for i in range(n2):
        line_v = d2[i, :]
        diff = d1 - line_v
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        ind_min1 = np.argmin(distances)
        min_dist1 = distances[ind_min1]

        distances[ind_min1] = np.inf

        ind_min2 = np.argmin(distances)
        min_dist2 = distances[ind_min2]

        if min_dist1 / min_dist2 < 0.5:
            possible_match_2 = cv.DMatch(i, ind_min1, min_dist1)
            for j in range(len(possible_matches_1)):
                match_found = (possible_match_2 == possible_matches_1[j])
                if match_found:
                    matches.append(possible_match_2)
                    break

    return matches


# Plain matching with cross-checking
def match_3(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    possible_matches_1 = []
    for i in range(n1):
        line_v = d1[i, :]
        diff = d2 - line_v
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        ind_min1 = np.argmin(distances)
        min_dist1 = distances[ind_min1]

        possible_matches_1.append(cv.DMatch(i, ind_min1, min_dist1))

    matches = []
    for i in range(n2):
        line_v = d2[i, :]
        diff = d1 - line_v
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        ind_min1 = np.argmin(distances)
        min_dist1 = distances[ind_min1]

        possible_match_2 = cv.DMatch(i, ind_min1, min_dist1)
        for j in range(len(possible_matches_1)):
            match_found = (possible_match_2 == possible_matches_1[j])
            if match_found:
                matches.append(possible_match_2)
                break

    return matches


def global_entropy(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _bins = 128

    hist, _ = np.histogram(gray.ravel(), bins=_bins, range=(0, _bins))

    prob_dist = hist / hist.sum()
    img_entropy = entropy(prob_dist, base=2)

    return img_entropy


def local_entropy(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    row, column = gray.shape

    img_local_entropy = (np.zeros([row, column])).astype(np.float64)

    _bins = 128
    limit_i = range(4, row-4)
    limit_j = range(4, column-4)

    for i in limit_i:
        for j in limit_j:
            window = gray[(i-4):(i+5), (j-4):(j+5)]

            hist, _ = np.histogram(window.ravel(), bins=_bins, range=(0, _bins))
            hist = hist[hist > 0]  # Ignore values of zero.
            prob_dist = hist / hist.sum()

            img_local_entropy[i, j] = entropy(prob_dist, base=2)

    img_local_entropy = img_local_entropy[4:(row - 4), 4:(column - 4)]

    return img_local_entropy
