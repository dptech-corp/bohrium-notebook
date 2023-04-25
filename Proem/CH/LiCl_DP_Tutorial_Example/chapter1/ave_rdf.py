import numpy as np
import matplotlib.pyplot as plt

nbins = 100 # define the number of bins in the RDF

with open("licl.rdf", "r") as f: # read the licl.rdf file
    lines = f.readlines()
    lines = lines[3:]

    data = np.zeros((nbins, 7))  
    count = 0  

    for line in lines:  
        nums = line.split()      
        if len(nums) == 8:  
            for i in range(1, 8):  
                data[int(nums[0])-1, i-1] += float(nums[i])  # accumulatie data for each bin  
        if len(nums) == 2:  
            count += 1         # count the number of accumulations for each bin
       
ave_rdf = data / count  # calculate the averaged RDF data
np.savetxt('ave_rdf.txt',ave_rdf)

labels = ['Li-Cl', 'Li-Li', 'Cl-Cl']
for i, label in zip(range(1, 7, 2), labels):
    plt.plot(ave_rdf[:, 0], ave_rdf[:, i], label=label)
plt.xlabel('r/Å')
plt.ylabel('g(r)')
plt.legend()
plt.savefig('rdf.png', dpi=300)