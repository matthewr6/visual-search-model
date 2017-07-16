S1RFSIZES = range(7,30,2)
# S1RFSIZES = range(3,26,2)
# S1RFSIZES = range(5,26,2)
NBS1SCALES = len(S1RFSIZES)
C1RFSIZE = 9


# Additive constants in the denominator for the normalizations in the S 
# stages.
SIGMAS = .5 
SIGMAS1 = 0
STRNORMLIP = 5

NBPROTS = 600
NBS3PROTS = 1720

NBKEPTWEIGHTS = 100

IMAGESFORPROTS = './naturalimages'
# IMAGESFOROBJPROTS = './objectimages'
IMAGESFOROBJPROTS = './gdrivesets/objs'

# GAUSSFACTOR = 150.0
# IORSIGMA = 35
# IORSIGMA = 17.5
# GAUSSFACTOR = 25.0
IORSIGMA = 15
# GAUSSFACTOR = 150.0
GAUSSFACTOR = 7.5

# 15 and 7.5
# try 25 for sigma?

# default is better than sigma 10 I think