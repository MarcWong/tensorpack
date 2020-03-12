# Author: Tao Hu <taohu620@gmail.com>
from tqdm import tqdm
with open("/Users/ht/Downloads/20180401_result.csv","w") as outfile:
    with open("/Users/ht/Downloads/20180401.csv","r") as f:
        lines = f.readlines()
        all_list = []
        to_be_removed = []
        for line in tqdm(lines[1:]):#remove first line
            line = line.strip("\r\n")
            line_list = line.split(",")
            if len(line_list) == 3:
                all_list.append(line_list[0])
                #print line_list[2]
                if line_list[2] != "":
                    to_be_removed.append(line_list[2])
            else:
                raise
        # write result
        all_set = set(all_list)
        to_be_removed_set =set(to_be_removed)
        result_set  = all_set - to_be_removed_set

        print len(result_set)
        for x in result_set:
            outfile.write("{}\r\n".format(x))
        """
        for number in tqdm(all_list):
            shouldDelete = False
            for tmp in to_be_removed:
                if number == tmp:
                    shouldDelete = True
                    break #to be remove
            if not shouldDelete:
                outfile.write("{}\n".format(number))
        """





