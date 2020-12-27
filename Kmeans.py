from numpy import array, dot, mean, argsort
import gensim
import random


def kmeans(tokenListInput, word2vecModel, numIters, numOfCentroids):
    # basic ideas: for each word in a sentence, we are going to translate it into a vector.
    # We define the centroid of a cluster to be the average of all vectors within the cluster.
    # notice that each element of a vector is a feature vector.
    def tokenListToVecArray(tokenl):
        v1 = []
        # print tokenList
        for word in tokenl:
            # each model[word] is a feature vector of size 100 (current default setting)
            # seems that gensim can sometimes miss some words... not sure how to fix
            if word not in word2vecModel.wv.vocab:
                v1.append([-0.9]*100) # smoothing
            else:
                v1.append(word2vecModel[word])
        return gensim.matutils.unitvec(array(v1).mean(axis=0))

    def converge(mus1, mus2):
        for i in range(len(mus1)):
            mu1 = mus1[i]
            mu2 = mus2[i]
            print('dot product is', dot(mu1, mu2))
            if abs(1.0 - dot(mu1, mu2)) > 0.001:
                return False
        return True

    # This step of kmeans provides a rough idea of how the sentences are distributed.
    # store similarity score in a matrix to avoid extra calculations
    centroidIndexes = random.sample(list(range(0, len(tokenListInput))), numOfCentroids)
    # centroidIndexes = [91, 43, 81, 125, 79, 149, 34, 172, 100, 137, 77, 46, 29, 54, 5, 38, 10, 2, 166, 118, 20, 33, 23, 67, 105, 12, 123, 159, 126, 120, 84, 102, 152, 87, 136, 115, 52, 0, 101, 37, 130, 147, 26, 157, 68, 9, 24, 114, 66, 31, 160, 28, 72, 142, 133]

    # print 'numOfCentroids = ', numOfCentroids
    # print centroidIndexes
    # print '>>>>>>>>>>'

    mus = list()
    for cent in centroidIndexes:
        print(tokenListInput[cent])
        # to increase computation efficiency and potentially memory, we store vectors instead of actual word lists
        mus.append(tokenListToVecArray(tokenListInput[cent]))

    iterations = numIters
    assignments = [None for _ in range(len(tokenListInput))]
    assignmentsIndexList = [None for _ in range(len(tokenListInput))]
    lastAssignmentIndexList = [None for _ in range(len(tokenListInput))]
    vectorList = [None for _ in range(len(tokenListInput))]

    for xiIndex in range(0, len(tokenListInput)):
        xi = tokenListInput[xiIndex]
        vectorList[xiIndex] = tokenListToVecArray(xi)

    for numIter in range(0, iterations):
        print("kmeans: in iteration ", numIter)
        for xiIndex in range(0, len(tokenListInput)):
            xi = tokenListInput[xiIndex]
            xiVec = vectorList[xiIndex]

            maxSimilarity = -2
            # assign every sentence to a closet one by their cosine difference
            for muIndex in range(0, len(mus)):
                mu = mus[muIndex]

                # print xiVec, mu
                cosSimilarity = dot(xiVec, mu)

                # print cosSimilarity, maxSimilarity

                if cosSimilarity > maxSimilarity:
                    assignmentsIndexList[xiIndex] = muIndex
                    assignments[xiIndex] = mu
                    maxSimilarity = cosSimilarity

        # calculate new centroids by averaging points in the cluster
        clusterCountList = [0 for _ in range(0, len(mus))]
        clusterSumList = [0 for _ in range(0, len(mus))]
        for xiIndex in range(0, len(tokenListInput)):
            xiVec = vectorList[xiIndex]
            clusterSumList[assignmentsIndexList[xiIndex]] += xiVec
            clusterCountList[assignmentsIndexList[xiIndex]] += 1

        # get and store new centroids
        newMus = [None for m in range(len(mus))]
        for muIndex in range(len(mus)):
            if clusterCountList[muIndex] == 0:
                newMus[muIndex] = 0
            else:
                newMus[muIndex] = clusterSumList[muIndex] / clusterCountList[muIndex]

        if lastAssignmentIndexList == assignmentsIndexList:
            break

        mus = list(newMus)
        lastAssignmentIndexList = list(assignmentsIndexList)
    return mus, assignmentsIndexList, assignments, vectorList


# a: the mean distance between a sample and all other points in the same class
# b: the mean distance between a sample and all other points in the next nearest cluster.
# Silhouette Coefficient s for this sample is given by:
# s = (b-a) / max(a,b)
def silhouetteScore(mus, assignmentsIndexList, vectorList):
    '''
    citation: http://www.sciencedirect.com/science/article/pii/0377042787901257

    For each point p, first find the average distance between p and all other points in the
    same cluster (this is a measure of cohesion, call it A). Then find the average distance
    between p and all points in the nearest cluster (this is a measure of separation from the
    closest other cluster, call it B). The silhouette coefficient for p is defined as the difference
     between B and A divided by the greater of the two (max(A,B)).

    We evaluate the cluster coefficient of each point and from this we can obtain the
    'overall' average cluster coefficient.

    Intuitively, we are trying to measure the space between clusters.
    If cluster cohesion is good (A is small) and cluster separation is good (B is large),
    the numerator will be large, etc.

    S(i) will be a good measure of how tight each cluster is coupled.
    '''

    # assignmentsIndexList: index = the ith sentence, value = centroid of that sentence
    # vectorList: vector representation of a sentence
    silhouetteScoreL = list()
    muResultList = [list() for _ in range(len(mus))]
    for sentenceID in range(len(vectorList)):
        sVec = vectorList[sentenceID]
        muInd = assignmentsIndexList[sentenceID]
        sMu = mus[muInd]
        sCosSimilarity = dot(sVec, sMu)
        muResultList[muInd].append((sCosSimilarity, sentenceID))

    for ptInd in range(len(assignmentsIndexList)):
        sVec = vectorList[ptInd]
        mu = mus[assignmentsIndexList[ptInd]]
        currDist = dot(sVec, mu)
        allDist = [dot(sVec, mu) for mu in mus]
        allDist.remove(currDist)

        nextClosestDist = 0
        nextClosestClusterInd = 0
        for distInd in range(len(allDist)):
            if allDist[distInd] > nextClosestDist:
                nextClosestDist = allDist[distInd]
                nextClosestClusterInd = distInd

        nextClosestPtsList = muResultList[nextClosestClusterInd]

        # because the more positive it is, the more similar the two words are...

        b = mean(array([dot(sVec, vectorList[sID]) for _, sID in nextClosestPtsList]))

        ptsSameLabelList = muResultList[assignmentsIndexList[ptInd]]
        allDistA = [dot(sVec, vectorList[sID]) for _, sID in ptsSameLabelList]
        maxA = max(allDistA)
        allDistA.remove(maxA)
        a = mean(allDistA)

        silhouetteScoreL.append((a-b) / max(a, b))
    return silhouetteScoreL

# # Filter out sentences that are the farthest to their centroids.
# muResultList = [list() for _ in xrange(len(musR))]
# for sentenceID in range(len(sentenceList)):
#     sVec = vectorListR[sentenceID]
#     muInd = assignmentsIndexListR[sentenceID]
#     sMu = musR[muInd]
#     sCosSimilarity = dot(sVec, sMu)
#     muResultList[muInd].append((sCosSimilarity, sentenceList[sentenceID]))
#
# for muListInd in range(len(muResultList)):
#     muResultList[muListInd] = sorted(muResultList[muListInd])
#
# print '\nanormalies for each centroid:'
# for val in muResultList:
#     print val[0]
#
# print '\ncentroids:'
# for val in muResultList:
#     print val[-1]

# starting from the given centroid:
