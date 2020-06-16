#This function makes a single path break on the path
def break_path(path):
    broken=[[],[],[]]
    if len(path)>3:
        for i in range(2,len(path)):
            for j in range(i):
                if path[j]==path[i]:
                    broken[0]=path[0:j]
                    broken[1]=path[j:i]
                    broken[2]=path[i:]
                    for k in range(len(broken)):
                        if len(broken[k])==1:
                            broken[k]=[]
                    return broken
    else:
        return 0
    return 0

#This function breaks each path in the list origianl_set into the path class real_set
def break_and_add_path(original_set,real_set):
    for path in original_set:
        while True:
            temp=break_path(path)
            if temp==0:
                if path!=[]:
                    real_set.add_path(path)
                break
            else:
                if temp[0]!=[]:
                    real_set.add_path(temp[0])
                if temp[1]!=[]:
                    real_set.add_path(temp[1])
                path=temp[2]
