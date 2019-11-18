import pandas 
import numpy as np
from math import sqrt
def printDivider():
    print()
    print("-------------------------------------------------------------")
    print()


def applyBasis(x , y ):
    basisVector = np.array([x**2, y**2, sqrt(2)*x*y ,sqrt(2)*x ,sqrt(2)*y,1])
    return basisVector

def BasisSVM(alpha,target, desc , query, W_o ):
    total = 0 
    i = 0
    for row in desc : 
        total += target[i] * alpha[i] * np.dot(row,query) + W_o
        i +=1 

    return total

def KernelSVM(alpha,target,desc,query,W_o):
    total = 0 
    i = 0
    for row in desc : 
        total += target[i] * alpha[i] * (np.dot(row,query)+1)**2 + W_o
        i +=1 

    return total


if __name__ == "__main__":
    # Load dataset
    url = "q2_Dataset.csv"
    names = ["Dose1", "Dose2", "Class"]
    dataset = pandas.read_csv(url, names=names)
    array = dataset.values


    descriptive = array[:,0:2]
    target = array[:,2]
    alpha = np.array([7.1655,6.9060,2.0033,6.1144,5.9538])
    W_o = 0.3074


    printDivider()

    print("Part i")
    #part i 
    #applying Basis Function on all 
    query = [0.9,-0.9]
    i =0
    basisMatrix = np.empty([5, 6], dtype=float) 
    for row in descriptive:
        vector = applyBasis(row[0],row[1])
        basisMatrix[i] = vector
        i +=1 

    basisQuery = applyBasis(query[0],query[1])
    classification = BasisSVM(alpha, target, basisMatrix, basisQuery,W_o)
    print("Basis SVM for the queryquery ", end ="")
    print(query, end="")   
    print(" is : ", end="")
    print(classification)
    if (classification <= -1 ):
        print("Therefore We classify it as -1 or 'Safe' ")
    else:
        print("There we classify it as +1 or 'Dangerous'")
    


    printDivider()


    print("Part ii")
    queryTwo = [0.22,0.16]
    classification = KernelSVM(alpha, target, descriptive,queryTwo, W_o )
    print("Kernel Trick SVM for the query ", end ="")
    print(queryTwo, end="")   
    print(" is : ", end="")
    print(classification)
    if (classification <= -1 ):
        print("Therefore We classify it as -1 or 'Safe' ")
    else:
        print("There we classify it as +1 or 'Dangerous'")
    



    printDivider()

    print("Part iii")

    classification = KernelSVM(alpha, target, descriptive,query, W_o )
    print("Kernel Trick SVM for the query ", end ="")
    print(query, end="")   
    print(" is : ", end="")
    print(classification)
    if (classification <= -1 ):
        print("Therefore We classify it as -1 or 'Safe' ")
    else:
        print("There we classify it as +1 or 'Dangerous'")
    
    printDivider()

    basisQueryTwo = applyBasis(queryTwo[0], queryTwo[1])
    classification = BasisSVM(alpha, target, basisMatrix, basisQueryTwo,W_o)
    print("Basis SVM for the queryquery ", end ="")
    print(queryTwo, end="")   
    print(" is : ", end="")
    print(classification)
    if (classification <= -1 ):
        print("Therefore We classify it as -1 or 'Safe' ")
    else:
        print("There we classify it as +1 or 'Dangerous'")


    printDivider()
    print("Part iv ")
    print("""
    The calculations for the Kernel Polynomial are Big O(n * m ) where : 
    n is the number of support vectors needed to loop through
    m is the number of features you must consider in the dot product
    **asumming squaring and addition and multiplication (excluding dot product) take constant time O(1)

    The calculations for Basis Function SVM are Big O(n*p) where : 
    n is the number of support vectors needed to loop through
    p is the number of features in the basis vector you must consider in the dot product
    **assuming addition and multiplication (excluding dot product) take constant time O(1)

    Thus, asymptotically the Kernel Polynomial will always require less calculations since m < p holds for all basis functions
    """)