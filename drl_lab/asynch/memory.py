import numpy as np
class Memory:
    def __init__(self,s_shape,a_size,r_size,maxSize = 100000):

        s_shape.insert(0,maxSize)
        self.colSize = (a_size+r_size+1) #+1 for done boolean
        #state storages
        self.stateStorage = np.empty(s_shape,dtype='float32')
        self.newStateStorage = np.empty(s_shape,dtype='float32')
        #
        self.storage = np.empty([maxSize,self.colSize],dtype='float32')
        self.currentRow = 0
        self.maxSize = maxSize
        self.s_size = s_shape
        self.a_size = a_size
        self.filledOnce = False
    def addData(self,s,a,s_new,r,done):
        #all_data = np.append(s,a)
        #all_data = np.append(all_data,s_new)
        self.stateStorage[self.currentRow][0:,0:,0:] = np.copy(s)
        self.newStateStorage[self.currentRow][0:,0:,0:] = np.copy(s_new)

        #print(all_data)
        all_data = np.append(a,r)
        all_data = np.append(all_data,done)
        
        self.storage[self.currentRow] = all_data
        self.currentRow += 1
        if self.currentRow == self.maxSize: # reset when full
            self.full()
    def full(self):
        self.currentRow = 0
        self.filledOnce = True
        print("memory full yo")
    def getBatch(self,batchSize=10):
        if self.filledOnce == False:
            choices = np.random.randint(0,self.currentRow , size=batchSize)
        else:
            choices = np.random.randint(0,self.maxSize , size=batchSize)
        
        return  {"state":self.stateStorage[choices],
                "action":self.storage[choices][0:,0:self.a_size],
                "new_state":self.newStateStorage[choices],
                "reward":self.storage[choices][0:,-2:-1],
                "done":self.storage[choices][0:,-1:]
                }