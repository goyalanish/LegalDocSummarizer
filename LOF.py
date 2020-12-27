from numpy import array, dot, mean, argsort


class LOF(object):
    '''
    citation: http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
    '''

    mus = list()
    assignmentsIndexList = list()
    vectorList = list()
    k = 0
    muResultList = list()

    def __init__(self, mus, assignmentsIndexList, vectorList, k):
        self.mus = mus
        self.assignmentsIndexList = assignmentsIndexList
        self.vectorList = vectorList
        self.k = k

        # preprocess to generate a cluster list
        self.muResultList = [list() for _ in range(len(mus))]
        for sentenceID in range(len(vectorList)):
            sVec = vectorList[sentenceID]
            muInd = assignmentsIndexList[sentenceID]
            # print assignmentsIndexList
            # print 'requested mu_Ind = ', muInd
            sMu = mus[muInd]
            sCosSimilarity = dot(sVec, sMu)
            self.muResultList[muInd].append((sCosSimilarity, sentenceID))

    def Nk(self, sId):
        # gets k nearest neighbors
        # to get the neighbors, need to remove sId from the cluster
        pt = self.vectorList[sId]
        clusterInd = self.assignmentsIndexList[sId]
        cluster = self.muResultList[clusterInd]
        neighborsID = [sId for _, sId in cluster]
        # neighborsID.remove(sId)
        allDists = [dot(pt, self.vectorList[neighbor]) for neighbor in neighborsID]

        # get the ID of k nearest neighbors
        # farthest -> nearest. so the k-th nearest neighbor is the first element.
        return argsort(allDists)[-self.k:]

    def kDist(self, sId):
        if len(self.Nk(sId)) == 0:
            return 0
        return dot(self.vectorList[self.Nk(sId)[0]], self.vectorList[sId])

    def reachabilityDistK(self, sIdA, sIdB):
        '''
        k is a parameter
        :param sIdA: A
        :param sIdB: B
        :return: reachability_distance_k = min(k-dist(B), distance(A,B))
        In our case, since bigger similarity means closer, and reachability_distance
        returns a farther one, we want to change max to min.
        '''

        kDistB = self.kDist(sIdB)
        return min(kDistB, dot(self.vectorList[sIdA], self.vectorList[sIdB]))

    def lrd(self, sIdA):
        '''
        This function gets the local reachability density of A
        :param sIdA:
        :param sIdB:
        :return: local reachability density of A (lrd)
        '''

        denom = sum([self.reachabilityDistK(sIdA, sIdB) for sIdB in self.Nk(sIdA)]) / len(self.Nk(sIdA))
        return 1.0 / denom

    # calculate LOF
    def calcLOF(self, sIdA):
        Nk_A = self.Nk(sIdA)

        # print "in LOF", 'sIdA = ', sIdA, ', other points = ', Nk_A
        return sum([self.lrd(sIdB) / self.lrd(sIdA) for sIdB in Nk_A]) / len(Nk_A)
